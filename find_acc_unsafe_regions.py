#!/usr/bin/env python3
"""Find ReLU policy regions in an ACC safe set that are unsafe after one step.

WNN cells are used only to discover activation patterns.  Each discovered
pattern is reconstructed as one global ReLU region, intersected directly with
the supplied convex safe polytope, and propagated through the discrete ACC
dynamics.  There is no fixed-point or continuous-time collision analysis.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")

import numpy as np
from scipy.spatial import ConvexHull
import torch
import torch.nn as nn
import torch.nn.functional as functional


SCRIPT_DIR = Path(__file__).resolve().parent
VERITEX_SRC = SCRIPT_DIR / "veritex" / "src"
if str(VERITEX_SRC) not in sys.path:
    sys.path.insert(0, str(VERITEX_SRC))

from veritex.networks.ffnn import FFNN  # noqa: E402
from veritex.sets.cubedomain import CubeDomain  # noqa: E402


@dataclass
class AffineRegion:
    pattern: np.ndarray
    vertices: np.ndarray
    hrep_A: np.ndarray
    hrep_b: np.ndarray
    action_C: np.ndarray
    action_d: float
    post_vertices: np.ndarray
    minimum_margin: float
    ia_margin_lower_bound: float
    unsafe: bool
    ia_certified_safe: bool

    @property
    def area(self) -> float:
        return polygon_area(self.vertices)


class ACCReLUPolicy(nn.Module):
    """A raw-observation ReLU actor with exact clipping and physical scaling."""

    def __init__(self, sizes: Sequence[int], action_scale: float):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(sizes[index], sizes[index + 1])
            for index in range(len(sizes) - 1)
        )
        self.action_scale = float(action_scale)

    def raw_output(self, observations: torch.Tensor) -> torch.Tensor:
        value = observations.reshape(-1, 2)
        for layer in self.layers[:-1]:
            value = functional.relu(layer(value))
        return self.layers[-1](value)

    def preactivations(self, observations: torch.Tensor) -> torch.Tensor:
        value = observations.reshape(-1, 2)
        values = []
        for layer in self.layers[:-1]:
            value = layer(value)
            values.append(value)
            value = functional.relu(value)
        raw = self.layers[-1](value)
        values.extend((raw + 1.0, raw - 1.0))
        return torch.cat(values, dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raw = self.raw_output(observations)
        clipped = -1.0 + functional.relu(raw + 1.0) - functional.relu(raw - 1.0)
        return self.action_scale * clipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="ReLU checkpoint or run directory")
    parser.add_argument("safe_grid", type=Path, help="safe_grid_cells.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-acceleration", type=float, default=100.0)
    parser.add_argument("--contract-margin", type=float, default=0.0)
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        metavar=("P_LO", "P_HI", "V_LO", "V_HI"),
        help=(
            "Restrict discovery and final polytopes to this rectangular region. "
            "By default the complete supplied safe polytope is analyzed."
        ),
    )
    parser.add_argument("--max-local-regions", type=int, default=2000)
    parser.add_argument("--max-split-depth", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument(
        "--max-discovery-cells",
        type=int,
        default=0,
        help="Testing only; 0 processes every discovery cell and gives a complete result.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    matches = sorted(path.glob("*.cleanrl_model"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one *.cleanrl_model in {path}, found {len(matches)}"
        )
    return matches[0]


def load_policy(path: Path, action_scale: float) -> tuple[ACCReLUPolicy, dict]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(
        payload.get("model_state_dict"), dict
    ):
        raise ValueError(f"Unsupported checkpoint structure: {path}")
    saved_args = dict(payload.get("args", {}))
    if bool(saved_args.get("use_wnn_actor", False)):
        raise ValueError("The policy checkpoint must contain a ReLU actor, not a WNN")
    if str(saved_args.get("mlp_activation", "")).lower() != "relu":
        raise ValueError("The actor must use ReLU hidden activations")
    if bool(saved_args.get("use_obs_norm", False)):
        raise ValueError(
            "Observation normalization must be disabled or folded into the network first"
        )

    state = payload["model_state_dict"]
    weights = []
    for key, value in state.items():
        if key.startswith("actor_mean.") and key.endswith(".weight"):
            weights.append((int(key.split(".")[1]), value))
    weights.sort(key=lambda item: item[0])
    if not weights:
        raise ValueError("Checkpoint has no actor_mean linear layers")
    sizes = [int(weights[0][1].shape[1])]
    sizes.extend(int(value.shape[0]) for _, value in weights)
    if sizes[0] != 2 or sizes[-1] != 1:
        raise ValueError(f"Expected an ACC actor shaped 2 -> ... -> 1, got {sizes}")

    model = ACCReLUPolicy(sizes, action_scale)
    with torch.no_grad():
        for target, (index, _) in zip(model.layers, weights):
            target.weight.copy_(state[f"actor_mean.{index}.weight"])
            target.bias.copy_(state[f"actor_mean.{index}.bias"])
    return model.eval(), {"sizes": sizes, "saved_args": saved_args}


def build_veritex_network(model: ACCReLUPolicy) -> nn.Sequential:
    modules = []
    for layer in model.layers[:-1]:
        modules.extend((copy.deepcopy(layer), nn.ReLU()))

    final = model.layers[-1]
    expanded = nn.Linear(final.in_features, 2)
    collapsed = nn.Linear(2, 1)
    with torch.no_grad():
        expanded.weight[0].copy_(final.weight[0])
        expanded.weight[1].copy_(final.weight[0])
        expanded.bias[0].copy_(final.bias[0] + 1.0)
        expanded.bias[1].copy_(final.bias[0] - 1.0)
        collapsed.weight.copy_(
            torch.tensor([[model.action_scale, -model.action_scale]])
        )
        collapsed.bias.fill_(-model.action_scale)
    modules.extend((expanded, nn.ReLU(), collapsed))
    sequential = nn.Sequential(*modules).eval()

    probe = torch.tensor(
        [[0.0, 0.0], [200.0, -200.0], [100.0, 100.0]], dtype=torch.float32
    )
    with torch.no_grad():
        error = float(torch.max(torch.abs(sequential(probe) - model(probe))))
    if error > 2e-4:
        raise RuntimeError(f"VeriTex conversion differs from actor by {error}")
    return sequential


def load_safe_grid(path: Path) -> dict:
    path = path.expanduser().resolve()
    with np.load(path) as archive:
        required = (
            "position_edges",
            "velocity_edges",
            "safe_polytope_A",
            "safe_polytope_b",
            "safe_polygon_vertices",
            "discovery_cells",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Safe-grid archive is missing {missing}")
        result = {key: np.asarray(archive[key]) for key in required}
    expected_shape = (
        len(result["position_edges"]) - 1,
        len(result["velocity_edges"]) - 1,
    )
    if result["discovery_cells"].shape != expected_shape:
        raise ValueError("discovery_cells does not match the supplied grid edges")
    if result["safe_polytope_A"].shape[1] != 2:
        raise ValueError("The safe polytope must be two-dimensional")
    return result


def restrict_safe_grid(safe_grid: dict, region, tolerance: float) -> dict:
    if region is None:
        return safe_grid
    p_lo, p_hi, v_lo, v_hi = map(float, region)
    if not np.all(np.isfinite(region)) or p_hi <= p_lo or v_hi <= v_lo:
        raise ValueError("--region requires finite increasing position and velocity bounds")

    region_A = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    region_b = np.array([-p_lo, p_hi, -v_lo, v_hi], dtype=np.float64)
    polygon = np.asarray(safe_grid["safe_polygon_vertices"], dtype=np.float64)
    for normal, bound in zip(region_A, region_b):
        polygon = clip_polygon(polygon, normal, float(bound), tolerance)
        if len(polygon) < 3:
            break
    if len(polygon) < 3 or polygon_area(polygon) <= tolerance:
        raise ValueError("--region has no positive-area intersection with the safe set")

    restricted = dict(safe_grid)
    restricted["safe_polytope_A"] = np.vstack(
        (safe_grid["safe_polytope_A"], region_A)
    )
    restricted["safe_polytope_b"] = np.concatenate(
        (safe_grid["safe_polytope_b"], region_b)
    )
    restricted["safe_polygon_vertices"] = polygon
    position_edges = safe_grid["position_edges"]
    velocity_edges = safe_grid["velocity_edges"]
    intersects_region = (
        (position_edges[1:, None] > p_lo)
        & (position_edges[:-1, None] < p_hi)
        & (velocity_edges[None, 1:] > v_lo)
        & (velocity_edges[None, :-1] < v_hi)
    )
    restricted["discovery_cells"] = (
        safe_grid["discovery_cells"].astype(bool) & intersects_region
    )
    return restricted


def linear_interval(
    weight: np.ndarray, bias: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(weight, 0.0)
    negative = np.minimum(weight, 0.0)
    return (
        positive @ lo + negative @ hi + bias,
        positive @ hi + negative @ lo + bias,
    )


def policy_interval(
    model: ACCReLUPolicy, lo: np.ndarray, hi: np.ndarray
) -> tuple[float, float]:
    lower = np.asarray(lo, dtype=np.float64)
    upper = np.asarray(hi, dtype=np.float64)
    for layer in model.layers[:-1]:
        weight = layer.weight.detach().cpu().numpy().astype(np.float64)
        bias = layer.bias.detach().cpu().numpy().astype(np.float64)
        lower, upper = linear_interval(weight, bias, lower, upper)
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
    final = model.layers[-1]
    lower, upper = linear_interval(
        final.weight.detach().cpu().numpy().astype(np.float64),
        final.bias.detach().cpu().numpy().astype(np.float64),
        lower,
        upper,
    )
    lower = np.clip(lower, -1.0, 1.0) * model.action_scale
    upper = np.clip(upper, -1.0, 1.0) * model.action_scale
    return float(lower[0]), float(upper[0])


def contract_margin_lower_bound(
    state_lo: np.ndarray, max_acceleration: float
) -> float:
    braking_distance_hi = max(-state_lo[1], 0.0) ** 2 / (
        2.0 * max_acceleration
    )
    return float(state_lo[0] - braking_distance_hi)


def box_is_interval_safe(
    model: ACCReLUPolicy,
    lo: np.ndarray,
    hi: np.ndarray,
    dt: float,
    max_acceleration: float,
    margin: float,
    tolerance: float,
) -> bool:
    action_lo, _ = policy_interval(model, lo, hi)
    post_lo = np.array(
        [
            lo[0] + dt * lo[1] + 0.5 * dt * dt * action_lo,
            lo[1] + dt * action_lo,
        ]
    )
    lower_bound = contract_margin_lower_bound(post_lo, max_acceleration)
    return lower_bound >= margin - tolerance


def enumerate_local_patterns(
    network: FFNN,
    model: ACCReLUPolicy,
    lo: np.ndarray,
    hi: np.ndarray,
    cap: int,
    safe_A: np.ndarray,
    safe_b: np.ndarray,
    tolerance: float,
) -> tuple[set[bytes], bool, int]:
    root = CubeDomain(lo.tolist(), hi.tolist()).to_FVIM()
    stack = [(root, -1, np.array([], dtype=np.int64))]
    patterns = set()
    explored = 0
    node_cap = max(cap * 8, cap + 1)
    while stack:
        next_states = network.compute_state(stack.pop())
        explored += 1
        if explored > node_cap or len(patterns) + len(stack) > cap * 2:
            return set(), True, explored
        if not next_states:
            continue
        if next_states[0][1] == network._num_layer - 1:
            reachable = next_states[0][0]
            vertices = np.unique(np.asarray(reachable.vertices), axis=0)
            if len(vertices) < 3:
                continue
            try:
                hull = ConvexHull(vertices)
            except Exception:
                continue
            if hull.volume <= 1e-12:
                continue
            intersection = vertices[hull.vertices]
            for normal, bound in zip(safe_A, safe_b):
                intersection = clip_polygon(
                    intersection, normal, float(bound), tolerance
                )
                if len(intersection) < 3:
                    break
            if len(intersection) < 3 or polygon_area(intersection) <= tolerance:
                continue
            center = intersection.mean(axis=0)
            patterns.add(pattern_key(model, center))
            if len(patterns) > cap:
                return set(), True, explored
        else:
            stack.extend(next_states)
    return patterns, False, explored


def pattern_key(model: ACCReLUPolicy, point: np.ndarray) -> bytes:
    with torch.no_grad():
        active = (
            model.preactivations(torch.as_tensor(point[None], dtype=torch.float32))
            > 0.0
        ).cpu().numpy().reshape(-1)
    packed = np.packbits(active.astype(np.uint8))
    return int(active.size).to_bytes(4, "little") + packed.tobytes()


def unpack_pattern(key: bytes) -> np.ndarray:
    size = int.from_bytes(key[:4], "little")
    return np.unpackbits(np.frombuffer(key[4:], dtype=np.uint8))[:size].astype(bool)


def discover_box_patterns(
    network: FFNN,
    model: ACCReLUPolicy,
    lo: np.ndarray,
    hi: np.ndarray,
    cap: int,
    max_depth: int,
    safe_A: np.ndarray,
    safe_b: np.ndarray,
    tolerance: float,
) -> tuple[set[bytes], int, int]:
    tasks = [(np.asarray(lo), np.asarray(hi), 0)]
    patterns = set()
    explored = 0
    subdivisions = 0
    while tasks:
        task_lo, task_hi, depth = tasks.pop()
        found, overflow, nodes = enumerate_local_patterns(
            network,
            model,
            task_lo,
            task_hi,
            cap,
            safe_A,
            safe_b,
            tolerance,
        )
        explored += nodes
        if not overflow:
            patterns.update(found)
            continue
        if depth >= max_depth:
            raise RuntimeError(
                "VeriTex exceeded --max-local-regions after the maximum split depth "
                f"in box {task_lo.tolist()} to {task_hi.tolist()}"
            )
        dimension = int(np.argmax(task_hi - task_lo))
        midpoint = 0.5 * (task_lo[dimension] + task_hi[dimension])
        left_hi = task_hi.copy()
        left_hi[dimension] = midpoint
        right_lo = task_lo.copy()
        right_lo[dimension] = midpoint
        tasks.extend(
            ((task_lo, left_hi, depth + 1), (right_lo, task_hi, depth + 1))
        )
        subdivisions += 1
    return patterns, explored, subdivisions


def clip_polygon(
    polygon: np.ndarray, normal: np.ndarray, offset: float, tolerance: float
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    clipped = []
    previous = polygon[-1]
    previous_value = float(normal @ previous - offset)
    previous_inside = previous_value <= tolerance
    for current in polygon:
        current_value = float(normal @ current - offset)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            direction = current - previous
            denominator = float(normal @ direction)
            if abs(denominator) > 1e-14:
                ratio = float((offset - normal @ previous) / denominator)
                clipped.append(previous + np.clip(ratio, 0.0, 1.0) * direction)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    if not clipped:
        return np.empty((0, 2), dtype=np.float64)
    points = np.asarray(clipped, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    if len(points) > 1:
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance
    points = points[keep]
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= tolerance:
        points = points[:-1]
    return points


def affine_region_hrep(
    model: ACCReLUPolicy, pattern: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    expected = sum(layer.out_features for layer in model.layers[:-1]) + 2
    if pattern.size != expected:
        raise ValueError(f"Expected {expected} activation bits, got {pattern.size}")
    matrix = np.eye(2, dtype=np.float64)
    offset = np.zeros(2, dtype=np.float64)
    constraints_A = []
    constraints_b = []
    cursor = 0
    for layer in model.layers[:-1]:
        weight = layer.weight.detach().cpu().numpy().astype(np.float64)
        bias = layer.bias.detach().cpu().numpy().astype(np.float64)
        pre_matrix = weight @ matrix
        pre_offset = weight @ offset + bias
        active = pattern[cursor : cursor + layer.out_features]
        for row, value, is_active in zip(pre_matrix, pre_offset, active):
            constraints_A.append(-row if is_active else row)
            constraints_b.append(value if is_active else -value)
        matrix = active[:, None] * pre_matrix
        offset = active * pre_offset
        cursor += layer.out_features

    final = model.layers[-1]
    raw_matrix = (
        final.weight.detach().cpu().numpy().astype(np.float64) @ matrix
    ).reshape(2)
    raw_offset = float(
        (
            final.weight.detach().cpu().numpy().astype(np.float64) @ offset
            + final.bias.detach().cpu().numpy().astype(np.float64)
        ).item()
    )
    clip_offsets = (raw_offset + 1.0, raw_offset - 1.0)
    clip_active = pattern[cursor : cursor + 2]
    for value, is_active in zip(clip_offsets, clip_active):
        constraints_A.append(-raw_matrix if is_active else raw_matrix)
        constraints_b.append(value if is_active else -value)

    coefficient = float(clip_active[0]) - float(clip_active[1])
    action_C = model.action_scale * coefficient * raw_matrix
    action_d = model.action_scale * (
        -1.0
        + float(clip_active[0]) * (raw_offset + 1.0)
        - float(clip_active[1]) * (raw_offset - 1.0)
    )
    return (
        np.asarray(constraints_A, dtype=np.float64),
        np.asarray(constraints_b, dtype=np.float64),
        action_C,
        float(action_d),
    )


def polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def reconstruct_and_classify(
    model: ACCReLUPolicy,
    key: bytes,
    safe_A: np.ndarray,
    safe_b: np.ndarray,
    safe_polygon: np.ndarray,
    dt: float,
    max_acceleration: float,
    margin: float,
    tolerance: float,
) -> AffineRegion | None:
    pattern = unpack_pattern(key)
    activation_A, activation_b, action_C, action_d = affine_region_hrep(
        model, pattern
    )
    vertices = np.asarray(safe_polygon, dtype=np.float64)
    for normal, bound in zip(activation_A, activation_b):
        vertices = clip_polygon(vertices, normal, float(bound), tolerance)
        if len(vertices) < 3:
            return None
    if polygon_area(vertices) <= tolerance:
        return None

    plant_input = np.array([0.5 * dt * dt, dt], dtype=np.float64)
    dynamics = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    closed_loop = dynamics + plant_input[:, None] * action_C[None, :]
    affine_offset = plant_input * action_d
    post_vertices = vertices @ closed_loop.T + affine_offset

    post_lo = post_vertices.min(axis=0)
    ia_lower_bound = contract_margin_lower_bound(post_lo, max_acceleration)
    ia_safe = ia_lower_bound >= margin - tolerance
    margins = post_vertices[:, 0] - np.maximum(-post_vertices[:, 1], 0.0) ** 2 / (
        2.0 * max_acceleration
    )
    minimum_margin = float(margins.min())
    unsafe = minimum_margin < margin - tolerance
    if ia_safe and unsafe:
        raise RuntimeError("Internal error: interval filter contradicted exact vertices")
    return AffineRegion(
        pattern=pattern,
        vertices=vertices,
        hrep_A=np.vstack((safe_A, activation_A)),
        hrep_b=np.concatenate((safe_b, activation_b)),
        action_C=action_C,
        action_d=action_d,
        post_vertices=post_vertices,
        minimum_margin=minimum_margin,
        ia_margin_lower_bound=ia_lower_bound,
        unsafe=unsafe,
        ia_certified_safe=ia_safe,
    )


def flatten_arrays(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for index, array in enumerate(arrays):
        offsets[index + 1] = offsets[index] + len(array)
    if not arrays or offsets[-1] == 0:
        width = arrays[0].shape[1] if arrays else 2
        return np.empty((0, width), dtype=np.float64), offsets
    return np.concatenate(arrays, axis=0), offsets


def save_archive(regions: Sequence[AffineRegion], path: Path) -> None:
    vertices, vertex_offsets = flatten_arrays([region.vertices for region in regions])
    post_vertices, post_offsets = flatten_arrays(
        [region.post_vertices for region in regions]
    )
    hrep_A, hrep_offsets = flatten_arrays([region.hrep_A for region in regions])
    hrep_b = (
        np.concatenate([region.hrep_b for region in regions])
        if regions
        else np.empty(0, dtype=np.float64)
    )
    digests = [
        hashlib.sha1(np.packbits(region.pattern).tobytes()).hexdigest()[:16]
        for region in regions
    ]
    np.savez_compressed(
        path,
        vertices=vertices,
        vertex_offsets=vertex_offsets,
        post_vertices=post_vertices,
        post_vertex_offsets=post_offsets,
        hrep_A=hrep_A,
        hrep_b=hrep_b,
        hrep_offsets=hrep_offsets,
        action_C=np.asarray([region.action_C for region in regions]),
        action_d=np.asarray([region.action_d for region in regions]),
        pattern_digest=np.asarray(digests),
        area=np.asarray([region.area for region in regions]),
        minimum_post_contract_margin=np.asarray(
            [region.minimum_margin for region in regions]
        ),
        ia_post_contract_margin_lower_bound=np.asarray(
            [region.ia_margin_lower_bound for region in regions]
        ),
        unsafe=np.asarray([region.unsafe for region in regions], dtype=bool),
        ia_certified_safe=np.asarray(
            [region.ia_certified_safe for region in regions], dtype=bool
        ),
    )


def plot_regions(
    regions: Sequence[AffineRegion],
    safe_grid: dict,
    output_path: Path,
    max_acceleration: float,
    margin: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    safe_polygon = safe_grid["safe_polygon_vertices"]
    position_limits = (
        float(safe_polygon[:, 0].min()),
        float(safe_polygon[:, 0].max()),
    )
    velocity_limits = (
        float(safe_grid["velocity_edges"][0]),
        float(safe_grid["velocity_edges"][-1]),
    )

    for axis, post in zip(axes, (False, True)):
        patches = []
        colors = []
        for region in regions:
            points = region.post_vertices if post else region.vertices
            patches.append(Polygon(points, closed=True))
            colors.append("#c9473d" if region.unsafe else "#8fb996")
        if patches:
            axis.add_collection(
                PatchCollection(
                    patches,
                    facecolor=colors,
                    edgecolor="#ffffff",
                    linewidth=0.25,
                    alpha=0.85,
                )
            )
        if not post:
            closed = np.vstack((safe_polygon, safe_polygon[0]))
            axis.plot(closed[:, 0], closed[:, 1], color="#202020", linewidth=1.3)
        velocity = np.linspace(velocity_limits[0], velocity_limits[1], 1000)
        boundary = (
            np.maximum(-velocity, 0.0) ** 2 / (2.0 * max_acceleration) + margin
        )
        axis.plot(boundary, velocity, "k--", linewidth=1.2)
        axis.set(
            xlim=position_limits,
            ylim=velocity_limits,
            xlabel="Relative position",
            ylabel="Relative velocity",
            title="Source affine regions" if not post else "One-step post-images",
        )
    figure.suptitle("Red regions contain a one-step ACC safety violation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    if args.dt <= 0.0 or args.max_acceleration <= 0.0:
        raise ValueError("--dt and --max-acceleration must be positive")
    if args.max_local_regions <= 0 or args.max_split_depth < 0:
        raise ValueError("Invalid VeriTex region or split limit")

    checkpoint = resolve_checkpoint(args.model)
    model, model_metadata = load_policy(checkpoint, args.max_acceleration)
    sequential = build_veritex_network(model)
    network = FFNN(sequential)
    safe_grid = load_safe_grid(args.safe_grid)
    safe_grid = restrict_safe_grid(safe_grid, args.region, args.tolerance)
    discovery_indices = np.argwhere(safe_grid["discovery_cells"].astype(bool))
    complete = args.max_discovery_cells == 0
    if args.max_discovery_cells > 0:
        discovery_indices = discovery_indices[: args.max_discovery_cells]

    position_edges = safe_grid["position_edges"]
    velocity_edges = safe_grid["velocity_edges"]
    patterns = set()
    interval_safe_cells = 0
    veritex_cells = 0
    explored_nodes = 0
    subdivisions = 0
    for count, (position_index, velocity_index) in enumerate(discovery_indices, 1):
        lo = np.array(
            [position_edges[position_index], velocity_edges[velocity_index]],
            dtype=np.float64,
        )
        hi = np.array(
            [position_edges[position_index + 1], velocity_edges[velocity_index + 1]],
            dtype=np.float64,
        )
        cell_intersection = np.array(
            [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
            dtype=np.float64,
        )
        for normal, bound in zip(
            safe_grid["safe_polytope_A"], safe_grid["safe_polytope_b"]
        ):
            cell_intersection = clip_polygon(
                cell_intersection, normal, float(bound), args.tolerance
            )
            if len(cell_intersection) < 3:
                break
        if (
            len(cell_intersection) < 3
            or polygon_area(cell_intersection) <= args.tolerance
        ):
            continue
        lo = cell_intersection.min(axis=0)
        hi = cell_intersection.max(axis=0)
        if box_is_interval_safe(
            model,
            lo,
            hi,
            args.dt,
            args.max_acceleration,
            args.contract_margin,
            args.tolerance,
        ):
            interval_safe_cells += 1
            continue
        found, nodes, splits = discover_box_patterns(
            network,
            model,
            lo,
            hi,
            args.max_local_regions,
            args.max_split_depth,
            safe_grid["safe_polytope_A"],
            safe_grid["safe_polytope_b"],
            args.tolerance,
        )
        patterns.update(found)
        veritex_cells += 1
        explored_nodes += nodes
        subdivisions += splits
        if args.report_every > 0 and count % args.report_every == 0:
            print(
                f"[{count}/{len(discovery_indices)}] patterns={len(patterns)} "
                f"IA-safe={interval_safe_cells} VeriTex-cells={veritex_cells}",
                flush=True,
            )

    regions = []
    for key in sorted(patterns):
        region = reconstruct_and_classify(
            model,
            key,
            safe_grid["safe_polytope_A"],
            safe_grid["safe_polytope_b"],
            safe_grid["safe_polygon_vertices"],
            args.dt,
            args.max_acceleration,
            args.contract_margin,
            args.tolerance,
        )
        if region is not None:
            regions.append(region)
    regions.sort(key=lambda region: (not region.unsafe, -region.area))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "one_step_affine_regions.npz"
    save_archive(regions, archive_path)
    unsafe_regions = [region for region in regions if region.unsafe]
    summary = {
        "format": "acc_relu_one_step_regions_v1",
        "semantics": (
            "Global ReLU activation regions intersected with the PWL input safe "
            "polytope and classified against the continuous braking invariant "
            "after one discrete ACC step."
        ),
        "complete": complete,
        "checkpoint": str(checkpoint),
        "safe_grid": str(args.safe_grid.expanduser().resolve()),
        "network_sizes": model_metadata["sizes"],
        "dt": args.dt,
        "max_acceleration": args.max_acceleration,
        "contract_margin": args.contract_margin,
        "analysis_region": list(args.region) if args.region is not None else None,
        "discovery_cells_available": int(
            np.count_nonzero(safe_grid["discovery_cells"])
        ),
        "discovery_cells_processed": int(len(discovery_indices)),
        "cells_certified_safe_by_interval_arithmetic": interval_safe_cells,
        "cells_sent_to_veritex": veritex_cells,
        "veritex_nodes_explored": explored_nodes,
        "adaptive_subdivisions": subdivisions,
        "candidate_activation_patterns": len(patterns),
        "nonempty_global_regions": len(regions),
        "global_regions_certified_safe_by_post_box": sum(
            region.ia_certified_safe for region in regions
        ),
        "one_step_safe_regions": len(regions) - len(unsafe_regions),
        "one_step_unsafe_regions": len(unsafe_regions),
        "unsafe_source_area": float(sum(region.area for region in unsafe_regions)),
    }
    summary_path = output_dir / "one_step_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plot:
        plot_regions(
            regions,
            safe_grid,
            output_dir / "one_step_affine_regions.png",
            args.max_acceleration,
            args.contract_margin,
        )
    print(json.dumps(summary, indent=2))
    print(f"Saved regions: {archive_path}")
    print(f"Saved summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
