#!/usr/bin/env python3
"""Build a conservative piecewise-linear ACC safe set from a WNN grid."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="A WNN .cleanrl_model or a run directory containing exactly one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: runs/reassure_safe_grid/<checkpoint stem>.",
    )
    parser.add_argument("--max-acceleration", type=float, default=100.0)
    parser.add_argument("--contract-margin", type=float, default=0.0)
    parser.add_argument("--position-bounds", type=float, nargs=2, default=(0.0, 200.0))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def checkpoint_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"Checkpoint path does not exist: {path}")
    matches = sorted(path.glob("*.cleanrl_model"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one *.cleanrl_model in {path}, found {len(matches)}."
        )
    return matches[0]


def uniform_thresholds(
    lower: np.ndarray, upper: np.ndarray, bits: int
) -> np.ndarray:
    lower_tensor = torch.as_tensor(lower, dtype=torch.float32)
    upper_tensor = torch.as_tensor(upper, dtype=torch.float32)
    indices = torch.arange(1, bits + 1, dtype=torch.float32)
    delta = (upper_tensor - lower_tensor) / (bits + 1)
    return (lower_tensor[:, None] + delta[:, None] * indices).numpy()


def load_wnn_grid(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    saved_args = checkpoint.get("args")
    state = checkpoint.get("model_state_dict")
    if not isinstance(saved_args, dict) or not isinstance(state, dict):
        raise ValueError("Expected a checkpoint with args and model_state_dict")
    if not saved_args.get("use_wnn_actor", False):
        raise ValueError("The supplied checkpoint is not a WNN actor checkpoint")

    try:
        lower = state["obs_min"].detach().cpu().numpy().astype(np.float64).reshape(-1)
        upper = state["obs_max"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    except KeyError as error:
        raise ValueError("WNN checkpoint is missing obs_min or obs_max") from error
    if lower.shape != (2,) or upper.shape != (2,):
        raise ValueError("ACC grid classification requires exactly two observations")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("WNN observation limits must be finite")
    if np.any(upper <= lower):
        raise ValueError("Every WNN observation maximum must exceed its minimum")

    threshold_payload = checkpoint.get("threshold_payload")
    if isinstance(threshold_payload, dict) and "thresholds" in threshold_payload:
        thresholds = np.asarray(threshold_payload["thresholds"], dtype=np.float64)
        threshold_source = "checkpoint.threshold_payload"
    else:
        thermometer = str(saved_args.get("wnn_thermometer", "uniform")).lower()
        if thermometer != "uniform":
            raise ValueError(
                "This checkpoint does not save thresholds and is not marked as uniform; "
                "its grid cannot be reconstructed exactly"
            )
        bits = int(saved_args["bits"])
        if bits <= 0:
            raise ValueError("WNN bits must be positive")
        thresholds = uniform_thresholds(lower, upper, bits)
        threshold_source = "reconstructed_uniform_from_checkpoint"

    if thresholds.ndim != 2 or thresholds.shape[0] != 2:
        raise ValueError(
            f"Expected WNN thresholds with shape [2, bits], got {thresholds.shape}"
        )

    edges = []
    for dimension in range(2):
        interior = thresholds[dimension]
        interior = interior[
            (interior > lower[dimension]) & (interior < upper[dimension])
        ]
        dimension_edges = np.unique(
            np.concatenate(([lower[dimension]], interior, [upper[dimension]]))
        )
        if dimension_edges.size < 2 or np.any(np.diff(dimension_edges) <= 0.0):
            raise ValueError(f"Invalid threshold grid in observation dimension {dimension}")
        edges.append(dimension_edges)

    return {
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "position_edges": edges[0],
        "velocity_edges": edges[1],
    }


def evaluate_pwl_boundary(
    velocity: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray
) -> np.ndarray:
    values = slopes[:, None] * np.asarray(velocity).reshape(1, -1) + intercepts[:, None]
    return values.max(axis=0)


def clip_polygon(
    polygon: np.ndarray, normal: np.ndarray, offset: float, tolerance: float = 1e-10
) -> np.ndarray:
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


def polygon_from_hrep(
    matrix: np.ndarray,
    offsets: np.ndarray,
    position_bounds: np.ndarray,
    velocity_bounds: np.ndarray,
) -> np.ndarray:
    p_lo, p_hi = position_bounds
    v_lo, v_hi = velocity_bounds
    polygon = np.array(
        [[p_lo, v_lo], [p_hi, v_lo], [p_hi, v_hi], [p_lo, v_hi]],
        dtype=np.float64,
    )
    for normal, offset in zip(matrix, offsets):
        polygon = clip_polygon(polygon, normal, float(offset))
        if len(polygon) == 0:
            raise ValueError("The piecewise-linear safe set is empty")
    return polygon


def polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def build_safe_polytope(
    grid: dict,
    max_acceleration: float,
    margin: float,
    position_bounds,
) -> dict:
    if max_acceleration <= 0.0:
        raise ValueError("--max-acceleration must be positive")

    position_edges = grid["position_edges"]
    velocity_edges = grid["velocity_edges"]
    position_bounds = np.asarray(position_bounds, dtype=np.float64)
    velocity_bounds = np.array([velocity_edges[0], velocity_edges[-1]])
    if position_bounds.shape != (2,) or position_bounds[1] <= position_bounds[0]:
        raise ValueError("--position-bounds requires increasing lower and upper bounds")
    if position_bounds[0] < position_edges[0] or position_bounds[1] > position_edges[-1]:
        raise ValueError("--position-bounds must lie inside the WNN position grid")

    negative_limit = min(0.0, velocity_bounds[1])
    negative_breakpoints = velocity_edges[
        (velocity_edges >= velocity_bounds[0]) & (velocity_edges <= negative_limit)
    ]
    breakpoints = np.unique(
        np.concatenate(
            ([velocity_bounds[0]], negative_breakpoints, [negative_limit])
        )
    )
    if len(breakpoints) < 2 and velocity_bounds[0] < 0.0:
        raise ValueError("At least two non-positive velocity breakpoints are required")

    exact_at_breakpoints = np.maximum(-breakpoints, 0.0) ** 2 / (
        2.0 * max_acceleration
    )
    if len(breakpoints) >= 2:
        slopes = np.diff(exact_at_breakpoints) / np.diff(breakpoints)
        intercepts = exact_at_breakpoints[:-1] - slopes * breakpoints[:-1]
    else:
        slopes = np.empty(0, dtype=np.float64)
        intercepts = np.empty(0, dtype=np.float64)
    # The zero segment represents the exact boundary for non-closing velocities.
    slopes = np.append(slopes, 0.0)
    intercepts = np.append(intercepts, 0.0)

    contract_matrix = np.column_stack((-np.ones_like(slopes), slopes))
    contract_offsets = -intercepts - margin
    domain_matrix = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    domain_offsets = np.array(
        [
            -position_bounds[0],
            position_bounds[1],
            -velocity_bounds[0],
            velocity_bounds[1],
        ],
        dtype=np.float64,
    )
    safe_matrix = np.vstack((contract_matrix, domain_matrix))
    safe_offsets = np.concatenate((contract_offsets, domain_offsets))
    safe_polygon = polygon_from_hrep(
        safe_matrix, safe_offsets, position_bounds, velocity_bounds
    )

    position_lower = position_edges[:-1, None]
    position_upper = position_edges[1:, None]
    velocity_lower = velocity_edges[None, :-1]
    velocity_upper = velocity_edges[None, 1:]
    shape = (
        len(position_edges) - 1,
        len(velocity_edges) - 1,
    )
    boundary_at_velocity_lower = evaluate_pwl_boundary(
        velocity_lower.reshape(-1), slopes, intercepts
    )[None, :] + margin
    boundary_at_velocity_upper = evaluate_pwl_boundary(
        velocity_upper.reshape(-1), slopes, intercepts
    )[None, :] + margin
    inside_cells = np.broadcast_to(
        (position_lower >= position_bounds[0])
        & (position_upper <= position_bounds[1])
        & (position_lower >= boundary_at_velocity_lower),
        shape,
    ).copy()
    intersects_safe_set = np.broadcast_to(
        (position_upper >= position_bounds[0])
        & (position_lower <= position_bounds[1])
        & (position_upper >= boundary_at_velocity_upper),
        shape,
    ).copy()
    boundary_cells = intersects_safe_set & ~inside_cells
    outside_cells = ~intersects_safe_set
    discovery_cells = intersects_safe_set

    position_widths = np.diff(position_edges)
    velocity_widths = np.diff(velocity_edges)
    cell_areas = position_widths[:, None] * velocity_widths[None, :]

    dense_velocity = np.linspace(velocity_bounds[0], velocity_bounds[1], 20001)
    exact_boundary = np.maximum(-dense_velocity, 0.0) ** 2 / (
        2.0 * max_acceleration
    ) + margin
    pwl_boundary = evaluate_pwl_boundary(
        dense_velocity, slopes, intercepts
    ) + margin
    approximation_gap = pwl_boundary - exact_boundary
    exact_width = np.maximum(position_bounds[1] - np.maximum(exact_boundary, position_bounds[0]), 0.0)
    exact_area = float(np.trapz(exact_width, dense_velocity))
    safe_area = polygon_area(safe_polygon)
    summary = {
        "format": "acc_wnn_pwl_safe_set_v1",
        "semantics": (
            "Convex inner approximation of the continuous ACC braking safe set, "
            "using secants at WNN velocity thresholds."
        ),
        "threshold_source": grid["threshold_source"],
        "thermometer_bits": int(grid["thresholds"].shape[1]),
        "grid_shape": list(shape),
        "position_bounds": position_bounds.tolist(),
        "velocity_bounds": velocity_bounds.tolist(),
        "max_acceleration": float(max_acceleration),
        "contract_margin": float(margin),
        "negative_velocity_breakpoints": int(len(breakpoints)),
        "linear_boundary_pieces": int(len(slopes)),
        "maximum_boundary_gap": float(approximation_gap.max()),
        "mean_boundary_gap": float(approximation_gap.mean()),
        "exact_safe_area": exact_area,
        "pwl_safe_area": safe_area,
        "pwl_to_exact_area_ratio": float(safe_area / exact_area),
        "total_grid_cells": int(np.prod(shape)),
        "inside_cells": int(inside_cells.sum()),
        "boundary_cells": int(boundary_cells.sum()),
        "outside_cells": int(outside_cells.sum()),
        "discovery_cells": int(discovery_cells.sum()),
        "inside_cell_area": float(cell_areas[inside_cells].sum()),
    }
    return {
        "summary": summary,
        "position_edges": position_edges,
        "velocity_edges": velocity_edges,
        "thresholds": grid["thresholds"],
        "velocity_breakpoints": breakpoints,
        "boundary_slopes": slopes,
        "boundary_intercepts": intercepts,
        "safe_polytope_A": safe_matrix,
        "safe_polytope_b": safe_offsets,
        "safe_polygon_vertices": safe_polygon,
        "inside_cells": inside_cells,
        "boundary_cells": boundary_cells,
        "outside_cells": outside_cells,
        "discovery_cells": discovery_cells,
    }


def plot_result(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = result["summary"]
    figure, axis = plt.subplots(figsize=(8.4, 6.4))
    velocity = np.linspace(*summary["velocity_bounds"], 2000)
    exact_position = np.maximum(-velocity, 0.0) ** 2 / (
        2.0 * summary["max_acceleration"]
    ) + summary["contract_margin"]
    pwl_position = evaluate_pwl_boundary(
        velocity, result["boundary_slopes"], result["boundary_intercepts"]
    ) + summary["contract_margin"]
    position_max = summary["position_bounds"][1]
    axis.fill_betweenx(
        velocity,
        np.minimum(pwl_position, position_max),
        position_max,
        color="#d7ead3",
        label="PWL safe set",
    )
    axis.fill_betweenx(
        velocity,
        np.minimum(exact_position, position_max),
        np.minimum(pwl_position, position_max),
        where=pwl_position >= exact_position,
        color="#e9b949",
        alpha=0.8,
        label="Conservative approximation gap",
    )
    axis.plot(
        exact_position,
        velocity,
        color="#111111",
        linestyle="--",
        linewidth=2.0,
        label="Exact braking boundary",
    )
    axis.plot(
        pwl_position,
        velocity,
        color="#287271",
        linewidth=1.8,
        label="Conservative PWL boundary",
    )
    axis.set(
        xlim=summary["position_bounds"],
        ylim=summary["velocity_bounds"],
        xlabel="Relative position",
        ylabel="Relative velocity",
        title="Piecewise-linear ACC safe-set approximation",
    )
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_result(result: dict, output_dir: Path, make_plot: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "safe_grid_summary.json"
    cells_path = output_dir / "safe_grid_cells.npz"
    summary_path.write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        cells_path,
        position_edges=result["position_edges"],
        velocity_edges=result["velocity_edges"],
        thresholds=result["thresholds"],
        velocity_breakpoints=result["velocity_breakpoints"],
        boundary_slopes=result["boundary_slopes"],
        boundary_intercepts=result["boundary_intercepts"],
        safe_polytope_A=result["safe_polytope_A"],
        safe_polytope_b=result["safe_polytope_b"],
        safe_polygon_vertices=result["safe_polygon_vertices"],
        inside_cells=result["inside_cells"],
        boundary_cells=result["boundary_cells"],
        outside_cells=result["outside_cells"],
        discovery_cells=result["discovery_cells"],
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved summary: {summary_path}")
    print(f"Saved cell masks: {cells_path}")
    if make_plot:
        plot_path = output_dir / "safe_grid.png"
        plot_result(result, plot_path)
        print(f"Saved plot: {plot_path}")


def main() -> None:
    args = parse_args()
    try:
        source = checkpoint_path(args.checkpoint)
        grid = load_wnn_grid(source)
        result = build_safe_polytope(
            grid,
            args.max_acceleration,
            args.contract_margin,
            args.position_bounds,
        )
        result["summary"]["checkpoint"] = str(source)
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path("runs/reassure_safe_grid") / source.stem
        save_result(result, output_dir.expanduser().resolve(), not args.no_plot)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
