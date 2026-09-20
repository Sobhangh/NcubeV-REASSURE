#!/usr/bin/env python3
"""Repair a ReLU ACC policy against a piecewise-linear one-step invariant.

Stock VeriTex properties combine an axis-aligned input box with polyhedral
unsafe output sets.  This driver conservatively turns each rectangular subset
of the ACC PWL safe set into an allowed interval for the policy acceleration.
The nonlinear braking invariant itself is not passed to VeriTex.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
VERITEX_SRC = SCRIPT_DIR / "veritex" / "src"
if str(VERITEX_SRC) not in sys.path:
    sys.path.insert(0, str(VERITEX_SRC))

from find_acc_unsafe_regions import (  # noqa: E402
    ACCReLUPolicy,
    build_veritex_network,
    load_policy,
    resolve_checkpoint,
)
from veritex.methods.repair import DATA, REPAIR  # noqa: E402
from veritex.methods.shared import SharedState  # noqa: E402
from veritex.methods.worker import Worker  # noqa: E402
from veritex.networks.ffnn import FFNN  # noqa: E402
from veritex.utils.sfproperty import Property  # noqa: E402


@dataclass(frozen=True)
class BoxConstraint:
    lower: np.ndarray
    upper: np.ndarray
    action_lower: float
    action_upper: float
    center_action: float

    @property
    def violation_score(self) -> float:
        return max(
            self.action_lower - self.center_action,
            self.center_action - self.action_upper,
            0.0,
        )


class ClampCorrection:
    """Move unsafe scalar outputs into a box's certified action interval."""

    def __init__(self, lower: float, upper: float, epsilon: float):
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon

    def __call__(self, unsafe_data):
        original_x = []
        corrected_y = []
        target_lower = self.lower + self.epsilon
        target_upper = self.upper - self.epsilon
        if target_lower > target_upper:
            midpoint = 0.5 * (self.lower + self.upper)
            target_lower = midpoint
            target_upper = midpoint
        for unsafe_x, unsafe_y in unsafe_data:
            x = torch.as_tensor(unsafe_x, dtype=torch.float32)
            y = torch.as_tensor(unsafe_y, dtype=torch.float32).clone()
            y.clamp_(min=target_lower, max=target_upper)
            original_x.append(x)
            corrected_y.append(y)
        return torch.cat(original_x, dim=0), torch.cat(corrected_y, dim=0)


class CappedWorkerRepair(REPAIR):
    """VeriTex repair with a scheduler-friendly worker cap."""

    def __init__(self, *args, num_workers: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers

    def compute_unsafe_data(self):
        self.ffnn = FFNN(self.torch_model, repair=True)
        all_unsafe_data = []
        for index, prop in enumerate(self.properties):
            logging.info("VeriTex property %d/%d", index + 1, len(self.properties))
            self.ffnn.set_property(prop)
            shared_state = SharedState(prop, self.num_workers)
            worker = Worker(self.ffnn, output_len=self.output_limit)
            processes = [
                mp.Process(target=worker.main_func, args=(worker_id, shared_state))
                for worker_id in range(self.num_workers)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            failed = [process.exitcode for process in processes if process.exitcode != 0]
            if failed:
                raise RuntimeError(f"VeriTex workers failed with exit codes {failed}")
            unsafe_data = [
                shared_state.outputs.get()
                for _ in range(shared_state.outputs_len.value)
            ]
            all_unsafe_data.append(unsafe_data)
        return all_unsafe_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="ReLU checkpoint or run directory")
    parser.add_argument("safe_grid", type=Path, help="safe_grid_cells.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/veritex_acc_repair"),
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--action-limit",
        type=float,
        default=150.0,
        help=(
            "Physical acceleration limit used by the converted controller. "
            "The original ACC experiments used 100."
        ),
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        metavar=("P_LO", "P_HI", "V_LO", "V_HI"),
        help="Restrict properties to a rectangular state region.",
    )
    parser.add_argument(
        "--max-properties",
        type=int,
        default=0,
        help="Keep the most visibly violated N boxes; 0 keeps every feasible box.",
    )
    parser.add_argument("--correction-epsilon", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-limit", type=int, default=100)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--preservation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_grid(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve()) as archive:
        required = (
            "position_edges",
            "velocity_edges",
            "safe_polytope_A",
            "safe_polytope_b",
            "discovery_cells",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Safe-grid archive is missing {missing}")
        return {name: np.asarray(archive[name]) for name in required}


def linear_maximum(
    coefficients: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    point = np.where(coefficients >= 0.0, upper, lower)
    return float(coefficients @ point)


def allowed_action_interval(
    lower: np.ndarray,
    upper: np.ndarray,
    safe_A: np.ndarray,
    safe_b: np.ndarray,
    dt: float,
    tolerance: float,
) -> tuple[float, float] | None:
    # post_state = state_matrix @ state + action_vector * acceleration
    state_matrix = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    action_vector = np.array([0.5 * dt * dt, dt], dtype=np.float64)
    action_lower = -np.inf
    action_upper = np.inf
    for normal, bound in zip(safe_A, safe_b):
        state_coefficients = normal @ state_matrix
        worst_state = linear_maximum(state_coefficients, lower, upper)
        action_coefficient = float(normal @ action_vector)
        residual = float(bound - worst_state)
        if action_coefficient > tolerance:
            action_upper = min(action_upper, residual / action_coefficient)
        elif action_coefficient < -tolerance:
            action_lower = max(action_lower, residual / action_coefficient)
        elif residual < -tolerance:
            return None
    return action_lower, action_upper


def rectangle_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [lower[0], lower[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
            [lower[0], upper[1]],
        ],
        dtype=np.float64,
    )


def clipped_cell_bounds(
    position_bounds: Sequence[float],
    velocity_bounds: Sequence[float],
    region: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    lower = np.array([position_bounds[0], velocity_bounds[0]], dtype=np.float64)
    upper = np.array([position_bounds[1], velocity_bounds[1]], dtype=np.float64)
    if region is not None:
        region_lower = np.array([region[0], region[2]], dtype=np.float64)
        region_upper = np.array([region[1], region[3]], dtype=np.float64)
        lower = np.maximum(lower, region_lower)
        upper = np.minimum(upper, region_upper)
    if np.any(upper - lower <= 0.0):
        return None
    return lower, upper


def build_constraints(
    model: ACCReLUPolicy,
    grid: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[BoxConstraint], dict[str, int]]:
    position_edges = grid["position_edges"]
    velocity_edges = grid["velocity_edges"]
    safe_A = np.asarray(grid["safe_polytope_A"], dtype=np.float64)
    safe_b = np.asarray(grid["safe_polytope_b"], dtype=np.float64)
    candidate_indices = np.argwhere(grid["discovery_cells"].astype(bool))
    constraints = []
    counts = {
        "candidate_grid_cells": int(len(candidate_indices)),
        "region_intersections": 0,
        "nonrectangular_or_outside": 0,
        "rectangular_safe_cells": 0,
        "dynamics_infeasible": 0,
        "control_infeasible": 0,
        "inactive_action_constraints": 0,
    }
    for position_index, velocity_index in candidate_indices:
        bounds = clipped_cell_bounds(
            position_edges[position_index : position_index + 2],
            velocity_edges[velocity_index : velocity_index + 2],
            args.region,
        )
        if bounds is None:
            continue
        counts["region_intersections"] += 1
        lower, upper = bounds
        corners = rectangle_corners(lower, upper)
        if np.any(safe_A @ corners.T - safe_b[:, None] > args.tolerance):
            counts["nonrectangular_or_outside"] += 1
            continue
        counts["rectangular_safe_cells"] += 1
        interval = allowed_action_interval(
            lower, upper, safe_A, safe_b, args.dt, args.tolerance
        )
        if interval is None or interval[0] > interval[1] + args.tolerance:
            counts["dynamics_infeasible"] += 1
            continue
        action_lower, action_upper = interval
        if (
            action_lower > args.action_limit + args.tolerance
            or action_upper < -args.action_limit - args.tolerance
        ):
            counts["control_infeasible"] += 1
            continue
        center = 0.5 * (lower + upper)
        with torch.no_grad():
            center_action = float(
                model(torch.as_tensor(center[None], dtype=torch.float32)).item()
            )
        constraints.append(
            BoxConstraint(lower, upper, action_lower, action_upper, center_action)
        )
    active_constraints = [
        item
        for item in constraints
        if item.action_lower > -args.action_limit
        or item.action_upper < args.action_limit
    ]
    counts["inactive_action_constraints"] = len(constraints) - len(active_constraints)
    constraints = active_constraints
    constraints.sort(key=lambda item: item.violation_score, reverse=True)
    if args.max_properties > 0:
        constraints = constraints[: args.max_properties]
    counts["selected_properties"] = len(constraints)
    counts["center_violations"] = sum(item.violation_score > 0.0 for item in constraints)
    return constraints, counts


def make_properties(
    constraints: Sequence[BoxConstraint],
    input_ranges: list[list[float]],
    action_limit: float,
    epsilon: float,
) -> list[list[object]]:
    properties_repair = []
    for constraint in constraints:
        unsafe_domains = []
        if constraint.action_lower > -action_limit:
            unsafe_domains.append(
                [
                    np.array([[1.0]], dtype=np.float64),
                    np.array([[-constraint.action_lower]], dtype=np.float64),
                ]
            )
        if constraint.action_upper < action_limit:
            unsafe_domains.append(
                [
                    np.array([[-1.0]], dtype=np.float64),
                    np.array([[constraint.action_upper]], dtype=np.float64),
                ]
            )
        if not unsafe_domains:
            continue
        prop = Property(
            [constraint.lower.tolist(), constraint.upper.tolist()],
            unsafe_domains,
            input_ranges=input_ranges,
        )
        correction = ClampCorrection(
            constraint.action_lower, constraint.action_upper, epsilon
        )
        properties_repair.append([prop, correction])
    return properties_repair


def freeze_clipping_head(model: nn.Sequential) -> list[nn.Parameter]:
    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    if len(linear_layers) < 3:
        raise ValueError("Expected hidden, clipping-expansion, and clipping-output layers")
    for layer in linear_layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable hidden-layer parameters remain")
    return trainable


def preservation_data(
    model: nn.Module,
    constraints: Sequence[BoxConstraint],
    sample_count: int,
    seed: int,
) -> DATA:
    if sample_count < 6:
        raise ValueError("--preservation-samples must be at least 6")
    rng = np.random.default_rng(seed)
    box_indices = rng.integers(0, len(constraints), size=sample_count)
    samples = np.empty((sample_count, 2), dtype=np.float32)
    for index, box_index in enumerate(box_indices):
        box = constraints[int(box_index)]
        samples[index] = rng.uniform(box.lower, box.upper)
    inputs = torch.from_numpy(samples)
    with torch.no_grad():
        outputs = model(inputs).detach().clone()
    first = int(0.6 * sample_count)
    second = int(0.8 * sample_count)
    return DATA(
        [inputs[:first], outputs[:first]],
        [inputs[first:second], outputs[first:second]],
        [inputs[second:], outputs[second:]],
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s - %(message)s",
    )


def run(args: argparse.Namespace) -> dict:
    if args.dt <= 0.0 or args.action_limit <= 0.0:
        raise ValueError("--dt and --action-limit must be positive")
    if args.workers <= 0 or args.output_limit <= 0:
        raise ValueError("--workers and --output-limit must be positive")
    if args.repair and args.verify_only:
        raise ValueError("Choose either --verify-only or --repair")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = resolve_checkpoint(args.model)
    policy, metadata = load_policy(checkpoint, args.action_limit)
    grid = load_grid(args.safe_grid)
    constraints, counts = build_constraints(policy, grid, args)
    if not constraints:
        raise ValueError("No feasible rectangular ACC properties were selected")

    input_ranges = [
        [float(grid["position_edges"][0]), float(grid["velocity_edges"][0])],
        [float(grid["position_edges"][-1]), float(grid["velocity_edges"][-1])],
    ]
    properties_repair = make_properties(
        constraints,
        input_ranges,
        args.action_limit,
        args.correction_epsilon,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "safe_grid": str(args.safe_grid.expanduser().resolve()),
        "network_sizes": metadata["sizes"],
        "dt": args.dt,
        "action_limit": args.action_limit,
        "region": args.region,
        **counts,
        "veritex_properties": len(properties_repair),
        "maximum_center_violation": max(
            item.violation_score for item in constraints
        ),
        "minimum_required_action": min(
            item.action_lower for item in constraints
        ),
        "maximum_required_action": max(
            item.action_lower for item in constraints
        ),
        "minimum_allowed_action_ceiling": min(
            item.action_upper for item in constraints
        ),
    }
    print(json.dumps(summary, indent=2))
    if not args.verify_only and not args.repair:
        return summary

    model = build_veritex_network(policy)
    trainable_parameters = freeze_clipping_head(model)
    data = preservation_data(
        model, constraints, args.preservation_samples, args.seed
    )
    repair = CappedWorkerRepair(
        model,
        properties_repair,
        data=data,
        output_limit=args.output_limit,
        num_workers=args.workers,
    )
    if args.verify_only:
        unsafe_data = repair.compute_unsafe_data()
        summary["properties_with_unsafe_data"] = sum(bool(data) for data in unsafe_data)
        summary["unsafe_samples"] = sum(len(data) for data in unsafe_data)
        print(json.dumps(summary, indent=2))
        return summary

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = optim.SGD(
        trainable_parameters, lr=args.learning_rate, momentum=0.9
    )
    repair.repair_model_regular(
        optimizer,
        nn.MSELoss(),
        args.alpha,
        args.beta,
        str(output_dir),
        iters=args.iterations,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    summary_path = output_dir / "repair_configuration.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    configure_logging()
    run(parse_args())


if __name__ == "__main__":
    main()
