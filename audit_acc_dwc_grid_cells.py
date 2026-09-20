#!/usr/bin/env python3
"""Audit native DWC grid cells against one-step ACC safety properties."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")

import numpy as np
import torch

from compute_acc_safe_grid import checkpoint_path, load_wnn_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="DWC checkpoint or run directory")
    parser.add_argument("safe_grid", type=Path, help="safe_grid_cells.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        default=(0.0, 200.0, -200.0, 0.0),
        metavar=("P_LO", "P_HI", "V_LO", "V_HI"),
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-acceleration", type=float, default=100.0)
    parser.add_argument("--contract-margin", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


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
    return np.asarray(clipped, dtype=np.float64).reshape(-1, 2)


def polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(
        0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )


class DWCPolicy:
    """Exact hard-forward evaluation of a trained DWC using its saved LUTs."""

    def __init__(self, checkpoint: Path):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = payload.get("args", {})
        state = payload.get("model_state_dict", {})
        if not args.get("use_wnn_actor", False):
            raise ValueError("The checkpoint is not a DWC/WNN actor")
        if bool(args.get("wnn_use_tanh_final", False)):
            raise ValueError("Tanh-final DWC checkpoints are not supported")

        grid = load_wnn_grid(checkpoint)
        self.thresholds = np.asarray(grid["thresholds"], dtype=np.float32)
        self.bits = int(args["bits"])
        self.widths = (
            [int(value) for value in args["sizes"]]
            if args.get("sizes") is not None
            else [int(args["size"])] * int(args["l"])
        )
        self.mappings = []
        self.luts = []
        for layer_index in range(1, len(self.widths) + 1):
            prefix = f"actor_mean.net.{layer_index}"
            luts = state[f"{prefix}.luts"].detach().cpu().numpy()
            weights_key = f"{prefix}.mapping.weights"
            if weights_key in state:
                winners = (
                    state[weights_key].detach().cpu().numpy().argmax(axis=0)
                )
                mapping = winners.reshape(luts.shape[0], -1)
            else:
                mapping = state[f"{prefix}.mapping"].detach().cpu().numpy()
            self.mappings.append(np.asarray(mapping, dtype=np.int64))
            self.luts.append(np.asarray(luts, dtype=np.float32))

        regression_index = len(self.widths) + 2
        self.log_alpha = (
            state[f"actor_mean.net.{regression_index}.log_alpha"]
            .detach()
            .cpu()
            .numpy()
        )
        self.beta = (
            state[f"actor_mean.net.{regression_index}.beta"]
            .detach()
            .cpu()
            .numpy()
        )
        self.metadata = {
            "seed": int(args.get("seed", -1)),
            "training_timesteps": int(args.get("total_timesteps", -1)),
            "bits": self.bits,
            "widths": self.widths,
            "lut_fan_in": int(args["n"]),
        }

    @staticmethod
    def _lut_layer(
        inputs: np.ndarray, mapping: np.ndarray, luts: np.ndarray
    ) -> np.ndarray:
        selected = inputs[:, mapping]
        powers = (1 << np.arange(mapping.shape[1], dtype=np.int64))[None, None]
        addresses = np.sum((selected > 0) * powers, axis=2, dtype=np.int64)
        values = luts[np.arange(luts.shape[0])[None, :], addresses]
        return (values > 0).astype(np.float32)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32).reshape(-1, 2)
        values = (states[:, :, None] > self.thresholds[None]).astype(np.float32)
        values = values.reshape(len(states), -1)
        for mapping, luts in zip(self.mappings, self.luts):
            values = self._lut_layer(values, mapping, luts)
        bucket_counts = values.sum(axis=1, keepdims=True)
        normalized = bucket_counts / float(values.shape[1])
        output = np.exp(self.log_alpha) * (normalized - 0.5) + self.beta
        return output.reshape(-1)


def load_safe_grid(path: Path) -> dict:
    with np.load(path.expanduser().resolve()) as archive:
        keys = (
            "position_edges",
            "velocity_edges",
            "safe_polytope_A",
            "safe_polytope_b",
            "safe_polygon_vertices",
        )
        return {key: np.asarray(archive[key]) for key in keys}


def cell_polygon(
    lo: np.ndarray,
    hi: np.ndarray,
    safe_A: np.ndarray,
    safe_b: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    polygon = np.array(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
        dtype=np.float64,
    )
    for normal, offset in zip(safe_A, safe_b):
        polygon = clip_polygon(polygon, normal, float(offset), tolerance)
        if len(polygon) < 3:
            break
    return polygon


def flatten_polygons(polygons: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(polygons) + 1, dtype=np.int64)
    for index, polygon in enumerate(polygons):
        offsets[index + 1] = offsets[index] + len(polygon)
    vertices = (
        np.concatenate(polygons, axis=0)
        if polygons
        else np.empty((0, 2), dtype=np.float64)
    )
    return vertices, offsets


def plot_results(
    path: Path,
    p_edges: np.ndarray,
    v_edges: np.ndarray,
    actions: np.ndarray,
    cells: list[dict],
    max_acceleration: float,
    margin: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    figure, axes = plt.subplots(1, 3, figsize=(17.2, 5.5), sharex=True, sharey=True)
    mesh = axes[0].pcolormesh(
        p_edges,
        v_edges,
        actions.T,
        shading="flat",
        cmap="RdBu_r",
        vmin=-max_acceleration,
        vmax=max_acceleration,
        rasterized=True,
    )
    colorbar = figure.colorbar(mesh, ax=axes[0], pad=0.02)
    colorbar.set_label("Physical relative acceleration")

    for axis, key, title in (
        (axes[1], "linear_unsafe", "Linearized one-step violation"),
        (axes[2], "nonlinear_unsafe", "Nonlinear one-step violation"),
    ):
        patches = [Polygon(cell["vertices"], closed=True) for cell in cells]
        colors = ["#c43d32" if cell[key] else "#83ad78" for cell in cells]
        axis.add_collection(
            PatchCollection(
                patches,
                facecolor=colors,
                edgecolor="#ffffff",
                linewidth=0.35,
            )
        )
        count = sum(cell[key] for cell in cells)
        axis.set_title(f"{title}\n{count}/{len(cells)} intersecting safe cells")

    velocity = np.linspace(v_edges[0], v_edges[-1], 1200)
    exact_boundary = np.maximum(-velocity, 0.0) ** 2 / (2.0 * max_acceleration) + margin
    for axis in axes:
        axis.plot(
            exact_boundary,
            velocity,
            color="#111111",
            linestyle="--",
            linewidth=1.7,
            label="Nonlinear braking boundary",
        )
        axis.set(
            xlim=(p_edges[0], p_edges[-1]),
            ylim=(v_edges[0], v_edges[-1]),
            xlabel="Relative position",
        )
        axis.legend(loc="lower right", fontsize=8)
    axes[0].set_title("DWC action on every native grid cell")
    axes[0].set_ylabel("Relative velocity")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    checkpoint = checkpoint_path(args.model)
    policy = DWCPolicy(checkpoint)
    safe_grid = load_safe_grid(args.safe_grid)
    p_lo, p_hi, v_lo, v_hi = (float(value) for value in args.box)
    if p_lo >= p_hi or v_lo >= v_hi:
        raise ValueError("--box bounds must be strictly increasing")

    all_p_edges = safe_grid["position_edges"]
    all_v_edges = safe_grid["velocity_edges"]
    p_mask = (all_p_edges >= p_lo - args.tolerance) & (
        all_p_edges <= p_hi + args.tolerance
    )
    v_mask = (all_v_edges >= v_lo - args.tolerance) & (
        all_v_edges <= v_hi + args.tolerance
    )
    p_edges = all_p_edges[p_mask]
    v_edges = all_v_edges[v_mask]
    if len(p_edges) < 2 or len(v_edges) < 2:
        raise ValueError("The box does not contain complete native grid cells")
    if abs(p_edges[0] - p_lo) > args.tolerance or abs(p_edges[-1] - p_hi) > args.tolerance:
        raise ValueError("Position box bounds must coincide with native grid edges")
    if abs(v_edges[0] - v_lo) > args.tolerance or abs(v_edges[-1] - v_hi) > args.tolerance:
        raise ValueError("Velocity box bounds must coincide with native grid edges")

    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    center_grid = np.stack(np.meshgrid(p_centers, v_centers, indexing="ij"), axis=-1)
    normalized_actions = policy(center_grid.reshape(-1, 2)).reshape(
        len(p_centers), len(v_centers)
    )
    actions = args.max_acceleration * np.clip(normalized_actions, -1.0, 1.0)

    safe_A = safe_grid["safe_polytope_A"]
    safe_b = safe_grid["safe_polytope_b"]
    cells = []
    for i in range(len(p_centers)):
        for j in range(len(v_centers)):
            lo = np.array([p_edges[i], v_edges[j]], dtype=np.float64)
            hi = np.array([p_edges[i + 1], v_edges[j + 1]], dtype=np.float64)
            vertices = cell_polygon(lo, hi, safe_A, safe_b, args.tolerance)
            if len(vertices) < 3 or polygon_area(vertices) <= args.tolerance:
                continue
            action = float(actions[i, j])
            post_vertices = np.column_stack(
                (
                    vertices[:, 0]
                    + args.dt * vertices[:, 1]
                    + 0.5 * args.dt**2 * action,
                    vertices[:, 1] + args.dt * action,
                )
            )
            linear_unsafe = bool(
                np.any(post_vertices @ safe_A.T - safe_b > args.tolerance)
            )
            closing = np.maximum(-post_vertices[:, 1], 0.0)
            margins = (
                post_vertices[:, 0]
                - closing**2 / (2.0 * args.max_acceleration)
                - args.contract_margin
            )
            cells.append(
                {
                    "indices": (i, j),
                    "lo": lo,
                    "hi": hi,
                    "vertices": vertices,
                    "post_vertices": post_vertices,
                    "action": action,
                    "linear_unsafe": linear_unsafe,
                    "nonlinear_unsafe": bool(margins.min() < -args.tolerance),
                    "minimum_margin": float(margins.min()),
                    "area": polygon_area(vertices),
                }
            )

    boundary_states = np.column_stack(
        (
            np.maximum(-v_centers, 0.0) ** 2
            / (2.0 * args.max_acceleration)
            + args.contract_margin
            + 10.0 * args.tolerance,
            v_centers,
        )
    )
    boundary_actions = args.max_acceleration * np.clip(
        policy(boundary_states), -1.0, 1.0
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_vertices, source_offsets = flatten_polygons(
        [cell["vertices"] for cell in cells]
    )
    post_vertices, post_offsets = flatten_polygons(
        [cell["post_vertices"] for cell in cells]
    )
    np.savez_compressed(
        output_dir / "dwc_grid_cell_audit.npz",
        position_edges=p_edges,
        velocity_edges=v_edges,
        normalized_action=normalized_actions,
        physical_action=actions,
        cell_indices=np.asarray([cell["indices"] for cell in cells], dtype=np.int64),
        cell_lo=np.asarray([cell["lo"] for cell in cells]),
        cell_hi=np.asarray([cell["hi"] for cell in cells]),
        source_vertices=source_vertices,
        source_vertex_offsets=source_offsets,
        post_vertices=post_vertices,
        post_vertex_offsets=post_offsets,
        cell_action=np.asarray([cell["action"] for cell in cells]),
        linear_unsafe=np.asarray([cell["linear_unsafe"] for cell in cells]),
        nonlinear_unsafe=np.asarray([cell["nonlinear_unsafe"] for cell in cells]),
        minimum_nonlinear_margin=np.asarray(
            [cell["minimum_margin"] for cell in cells]
        ),
        cell_safe_intersection_area=np.asarray([cell["area"] for cell in cells]),
        boundary_states=boundary_states,
        boundary_actions=boundary_actions,
    )

    linear_count = sum(cell["linear_unsafe"] for cell in cells)
    nonlinear_count = sum(cell["nonlinear_unsafe"] for cell in cells)
    summary = {
        "format": "acc_dwc_grid_cell_audit_v1",
        "complete": True,
        "checkpoint": str(checkpoint),
        "safe_grid": str(args.safe_grid.expanduser().resolve()),
        "model": policy.metadata,
        "query_box": list(args.box),
        "input_domain_semantics": "Native grid cells intersected with the PWL ACC input-safe polytope.",
        "policy_semantics": "Exact constant hard-DWC action per native thermometer cell, clipped to [-1, 1].",
        "dt": args.dt,
        "max_acceleration": args.max_acceleration,
        "contract_margin": args.contract_margin,
        "native_grid_shape_in_box": [len(p_centers), len(v_centers)],
        "native_grid_cells_in_box": int(len(p_centers) * len(v_centers)),
        "input_safe_intersecting_cells": len(cells),
        "linearized_one_step_violating_cells": linear_count,
        "nonlinear_one_step_violating_cells": nonlinear_count,
        "minimum_nonlinear_post_margin": float(
            min(cell["minimum_margin"] for cell in cells)
        ),
        "physical_action_range_in_box": [float(actions.min()), float(actions.max())],
        "boundary_probe_count": int(len(boundary_actions)),
        "boundary_full_braking_cells_at_least_99": int(
            np.count_nonzero(boundary_actions >= 99.0)
        ),
        "boundary_action_range": [
            float(boundary_actions.min()),
            float(boundary_actions.max()),
        ],
        "boundary_action_mean": float(boundary_actions.mean()),
        "artifacts": {
            "archive": "dwc_grid_cell_audit.npz",
            "plot": "dwc_grid_cell_audit.png" if not args.no_plot else None,
        },
    }
    (output_dir / "dwc_grid_cell_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not args.no_plot:
        plot_results(
            output_dir / "dwc_grid_cell_audit.png",
            p_edges,
            v_edges,
            actions,
            cells,
            args.max_acceleration,
            args.contract_margin,
        )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    try:
        run(parse_args())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
