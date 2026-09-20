#!/usr/bin/env python3
"""Exhaustively extract unsafe affine ReLU regions from one ACC input box.

The input box is intersected with the supplied piecewise-linear ACC safe set.
VeriTex then enumerates every ReLU activation pattern on that domain.  For the
linearized post-state invariant, this script returns an exact, disjoint union
of convex counterexample polygons.  For the nonlinear braking invariant, it
exactly identifies every affine region containing a violation and stores the
region's affine closed-loop map; its curved unsafe subset is not polygonized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")


SCRIPT_DIR = Path(__file__).resolve().parent
VERITEX_SRC = SCRIPT_DIR / "veritex" / "src"
if str(VERITEX_SRC) not in sys.path:
    sys.path.insert(0, str(VERITEX_SRC))

from find_acc_unsafe_regions import (  # noqa: E402
    affine_region_hrep,
    build_veritex_network,
    clip_polygon,
    discover_box_patterns,
    load_policy,
    load_safe_grid,
    polygon_area,
    resolve_checkpoint,
    restrict_safe_grid,
    unpack_pattern,
)
from veritex.networks.ffnn import FFNN  # noqa: E402


@dataclass
class AffineRegion:
    pattern: np.ndarray
    vertices: np.ndarray
    action_C: np.ndarray
    action_d: float
    post_matrix: np.ndarray
    post_offset: np.ndarray
    post_vertices: np.ndarray
    linear_unsafe: bool
    nonlinear_unsafe: bool
    nonlinear_minimum_margin: float


@dataclass
class CounterexamplePiece:
    affine_region_index: int
    violated_constraint_index: int
    vertices: np.ndarray
    post_vertices: np.ndarray


@dataclass
class Rollout:
    start_kind: str
    states: np.ndarray
    actions: np.ndarray
    margins: np.ndarray
    first_violation_step: int
    first_violation_reason: str
    termination_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="ReLU checkpoint or run directory")
    parser.add_argument("safe_grid", type=Path, help="safe_grid_cells.npz")
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        required=True,
        metavar=("P_LO", "P_HI", "V_LO", "V_HI"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--property",
        choices=("linearized", "nonlinear", "both"),
        default="both",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=100.0,
        help="Physical acceleration represented by normalized action 1.",
    )
    parser.add_argument(
        "--braking-acceleration",
        type=float,
        default=100.0,
        help="Acceleration in the nonlinear braking-distance invariant.",
    )
    parser.add_argument("--contract-margin", type=float, default=0.0)
    parser.add_argument("--max-local-regions", type=int, default=2000)
    parser.add_argument("--max-split-depth", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--rollouts-per-kind", type=int, default=5)
    parser.add_argument("--rollout-inside-offset", type=float, default=1.0)
    parser.add_argument("--rollout-tolerance", type=float, default=1e-5)
    parser.add_argument("--no-rollouts", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def flatten_arrays(
    arrays: Sequence[np.ndarray], width: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for index, array in enumerate(arrays):
        offsets[index + 1] = offsets[index] + len(array)
    if not arrays or offsets[-1] == 0:
        return np.empty((0, width), dtype=np.float64), offsets
    return np.concatenate(arrays, axis=0), offsets


def flatten_vectors(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for index, array in enumerate(arrays):
        offsets[index + 1] = offsets[index] + len(array)
    if not arrays or offsets[-1] == 0:
        return np.empty(0, dtype=np.float64), offsets
    return np.concatenate(arrays), offsets


def reconstruct_region(
    model,
    pattern_key: bytes,
    input_polygon: np.ndarray,
    dt: float,
    tolerance: float,
) -> AffineRegion | None:
    pattern = unpack_pattern(pattern_key)
    activation_A, activation_b, action_C, action_d = affine_region_hrep(
        model, pattern
    )
    vertices = np.asarray(input_polygon, dtype=np.float64)
    for normal, bound in zip(activation_A, activation_b):
        vertices = clip_polygon(vertices, normal, float(bound), tolerance)
        if len(vertices) < 3:
            return None
    if polygon_area(vertices) <= tolerance:
        return None

    plant_input = np.array([0.5 * dt * dt, dt], dtype=np.float64)
    dynamics = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    post_matrix = dynamics + plant_input[:, None] * action_C[None, :]
    post_offset = plant_input * action_d
    post_vertices = vertices @ post_matrix.T + post_offset
    return AffineRegion(
        pattern=pattern,
        vertices=vertices,
        action_C=action_C,
        action_d=action_d,
        post_matrix=post_matrix,
        post_offset=post_offset,
        post_vertices=post_vertices,
        linear_unsafe=False,
        nonlinear_unsafe=False,
        nonlinear_minimum_margin=np.nan,
    )


def extract_linear_counterexamples(
    region_index: int,
    region: AffineRegion,
    safe_A: np.ndarray,
    safe_b: np.ndarray,
    tolerance: float,
) -> list[CounterexamplePiece]:
    """Partition R \\ post_safe_set into disjoint convex source polygons."""
    remaining = region.vertices
    pieces = []
    for constraint_index, (normal, bound) in enumerate(zip(safe_A, safe_b)):
        source_normal = normal @ region.post_matrix
        source_bound = float(bound - normal @ region.post_offset)

        # Allocate every point violating this constraint before moving on to
        # the next one. This makes the resulting union disjoint up to borders.
        unsafe = clip_polygon(
            remaining, -source_normal, -source_bound, tolerance
        )
        if len(unsafe) >= 3 and polygon_area(unsafe) > tolerance:
            pieces.append(
                CounterexamplePiece(
                    affine_region_index=region_index,
                    violated_constraint_index=constraint_index,
                    vertices=unsafe,
                    post_vertices=(
                        unsafe @ region.post_matrix.T + region.post_offset
                    ),
                )
            )

        remaining = clip_polygon(
            remaining, source_normal, source_bound, tolerance
        )
        if len(remaining) < 3 or polygon_area(remaining) <= tolerance:
            break
    return pieces


def nonlinear_minimum_margin(
    post_vertices: np.ndarray, braking_acceleration: float
) -> float:
    closing_speed = np.maximum(-post_vertices[:, 1], 0.0)
    margins = post_vertices[:, 0] - closing_speed**2 / (
        2.0 * braking_acceleration
    )
    return float(margins.min())


def pattern_digest(pattern: np.ndarray) -> str:
    return hashlib.sha1(np.packbits(pattern).tobytes()).hexdigest()[:16]


def make_rollout_starts(
    query_box: Sequence[float],
    braking_acceleration: float,
    contract_margin: float,
    inside_offset: float,
    count: int,
) -> list[tuple[str, np.ndarray]]:
    p_lo, p_hi, v_lo, v_hi = query_box
    velocity_hi = min(v_hi, 0.0)
    candidate_velocity = np.linspace(v_lo, velocity_hi, 20001)
    boundary_position = (
        np.maximum(-candidate_velocity, 0.0) ** 2
        / (2.0 * braking_acceleration)
        + contract_margin
    )
    valid = (
        (boundary_position >= p_lo)
        & (boundary_position + inside_offset <= p_hi)
    )
    candidate_velocity = candidate_velocity[valid]
    boundary_position = boundary_position[valid]
    if len(candidate_velocity) < count:
        raise ValueError(
            "The query box does not contain enough nonlinear-boundary points "
            "for the requested rollouts"
        )
    # Concentrate starts near the largest closing speeds, where the one-step
    # counterexamples occur, while retaining coverage of the rest of the curve.
    fractions = np.linspace(0.0, 1.0, count + 2)[1:-1] ** 3
    sample_indices = np.rint(fractions * (len(candidate_velocity) - 1)).astype(
        np.int64
    )
    starts = []
    for index in sample_indices:
        boundary = np.array(
            [boundary_position[index], candidate_velocity[index]], dtype=np.float64
        )
        starts.append(("boundary", boundary))
        starts.append(("inside", boundary + np.array([inside_offset, 0.0])))
    return starts


def simulate_rollout(
    model,
    start_kind: str,
    initial_state: np.ndarray,
    steps: int,
    dt: float,
    action_limit: float,
    braking_acceleration: float,
    contract_margin: float,
    maximum_position: float,
    tolerance: float,
) -> Rollout:
    states = [np.asarray(initial_state, dtype=np.float64)]
    actions = []
    margins = []
    first_violation_step = -1
    first_violation_reason = ""
    for step in range(1, steps + 1):
        state = states[-1]
        with torch.no_grad():
            action = float(
                model(torch.as_tensor(state[None], dtype=torch.float32)).item()
            )
        action = float(np.clip(action, -action_limit, action_limit))
        next_state = np.array(
            [
                state[0] + dt * state[1] + 0.5 * dt * dt * action,
                state[1] + dt * action,
            ],
            dtype=np.float64,
        )
        closing_speed = max(-next_state[1], 0.0)
        margin = (
            next_state[0]
            - closing_speed**2 / (2.0 * braking_acceleration)
            - contract_margin
        )
        tube_minimum_position = min(state[0], next_state[0])
        if abs(action) > np.finfo(np.float64).eps:
            critical_time = -state[1] / action
            if 0.0 < critical_time < dt:
                critical_position = (
                    state[0]
                    + critical_time * state[1]
                    + 0.5 * critical_time**2 * action
                )
                tube_minimum_position = min(
                    tube_minimum_position, critical_position
                )
        if first_violation_step < 0:
            if tube_minimum_position < -tolerance:
                first_violation_step = step
                first_violation_reason = "continuous_collision"
            elif margin < -tolerance:
                first_violation_step = step
                first_violation_reason = "nonlinear_endpoint"
            elif next_state[0] > maximum_position + tolerance:
                first_violation_step = step
                first_violation_reason = "position_above_maximum"
        states.append(next_state)
        actions.append(action)
        margins.append(margin)
        if tube_minimum_position < -tolerance:
            termination_reason = "collision"
            break
        if next_state[0] > maximum_position + tolerance:
            termination_reason = "position_above_maximum"
            break
    else:
        termination_reason = "time_limit"
    return Rollout(
        start_kind=start_kind,
        states=np.asarray(states),
        actions=np.asarray(actions),
        margins=np.asarray(margins),
        first_violation_step=first_violation_step,
        first_violation_reason=first_violation_reason,
        termination_reason=termination_reason,
    )


def closed_polygon(vertices: np.ndarray) -> np.ndarray:
    if len(vertices) == 0:
        return vertices
    return np.vstack((vertices, vertices[0]))


def set_local_limits(axis, arrays: Sequence[np.ndarray]) -> None:
    points = np.concatenate([array for array in arrays if len(array)], axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    padding = np.maximum(0.05 * (upper - lower), 1e-3)
    axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])


def plot_results(
    path: Path,
    query_box: Sequence[float],
    input_polygon: np.ndarray,
    post_safe_polygon: np.ndarray,
    regions: Sequence[AffineRegion],
    pieces: Sequence[CounterexamplePiece],
    evaluate_linear: bool,
    evaluate_nonlinear: bool,
    braking_acceleration: float,
    contract_margin: float,
    rollouts: Sequence[Rollout],
    rollout_steps: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    panel_count = 3 if rollouts else 2
    figure, axes = plt.subplots(
        1, panel_count, figsize=(6 * panel_count, 5), constrained_layout=True
    )
    source_axis, post_axis = axes[:2]
    region_color = "#52616b"
    input_color = "#d9e2e8"
    linear_color = "#c63d2f"
    nonlinear_color = "#e09f3e"
    safe_color = "#287a5b"

    source_axis.fill(
        input_polygon[:, 0],
        input_polygon[:, 1],
        color=input_color,
        alpha=0.65,
        zorder=0,
    )
    for region in regions:
        polygon = closed_polygon(region.vertices)
        if evaluate_nonlinear and region.nonlinear_unsafe:
            source_axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                color=nonlinear_color,
                alpha=0.22,
                zorder=1,
            )
        source_axis.plot(
            polygon[:, 0], polygon[:, 1], color=region_color, linewidth=0.8, zorder=2
        )
    if evaluate_linear:
        for piece in pieces:
            polygon = closed_polygon(piece.vertices)
            source_axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                color=linear_color,
                alpha=0.72,
                zorder=3,
            )
            source_axis.plot(
                polygon[:, 0],
                polygon[:, 1],
                color="#7d241d",
                linewidth=0.8,
                zorder=4,
            )
    p_lo, p_hi, v_lo, v_hi = query_box
    box = np.array(
        [[p_lo, v_lo], [p_hi, v_lo], [p_hi, v_hi], [p_lo, v_hi]],
        dtype=np.float64,
    )
    box = closed_polygon(box)
    source_axis.plot(
        box[:, 0], box[:, 1], color="#222222", linestyle="--", linewidth=1.0
    )
    set_local_limits(source_axis, [box, input_polygon])
    source_axis.set_title("Source-state regions")

    post_arrays = [region.post_vertices for region in regions]
    for region in regions:
        polygon = closed_polygon(region.post_vertices)
        if evaluate_nonlinear and region.nonlinear_unsafe:
            post_axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                color=nonlinear_color,
                alpha=0.22,
                zorder=1,
            )
        post_axis.plot(
            polygon[:, 0], polygon[:, 1], color=region_color, linewidth=0.8, zorder=2
        )
    if evaluate_linear:
        for piece in pieces:
            polygon = closed_polygon(piece.post_vertices)
            post_axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                color=linear_color,
                alpha=0.72,
                zorder=3,
            )
            post_axis.plot(
                polygon[:, 0],
                polygon[:, 1],
                color="#7d241d",
                linewidth=0.8,
                zorder=4,
            )
    safe_polygon = closed_polygon(post_safe_polygon)
    post_axis.plot(
        safe_polygon[:, 0],
        safe_polygon[:, 1],
        color=safe_color,
        linewidth=1.3,
        label="PWL safe boundary",
        zorder=5,
    )
    all_post_points = np.concatenate(post_arrays, axis=0)
    velocity = np.linspace(
        all_post_points[:, 1].min(), all_post_points[:, 1].max(), 500
    )
    nonlinear_position = (
        np.maximum(-velocity, 0.0) ** 2 / (2.0 * braking_acceleration)
        + contract_margin
    )
    post_axis.plot(
        nonlinear_position,
        velocity,
        color="#111111",
        linestyle=":",
        linewidth=1.4,
        label="Nonlinear braking boundary",
        zorder=6,
    )
    set_local_limits(post_axis, post_arrays)
    post_axis.set_title("One-step post-state regions")

    if rollouts:
        rollout_axis = axes[2]
        rollout_colors = {"boundary": "#245f9e", "inside": "#7b4f9d"}
        trajectory_arrays = []
        for rollout in rollouts:
            trajectory_arrays.append(rollout.states)
            color = rollout_colors[rollout.start_kind]
            rollout_axis.plot(
                rollout.states[:, 0],
                rollout.states[:, 1],
                color=color,
                linewidth=1.0,
                alpha=0.8,
                zorder=2,
            )
            marker = "o" if rollout.start_kind == "boundary" else "^"
            rollout_axis.scatter(
                rollout.states[0, 0],
                rollout.states[0, 1],
                color=color,
                marker=marker,
                s=28,
                zorder=4,
            )
            rollout_axis.scatter(
                rollout.states[-1, 0],
                rollout.states[-1, 1],
                facecolors="none",
                edgecolors=color,
                marker="s",
                s=26,
                zorder=4,
            )
            if rollout.first_violation_step >= 0:
                violation = rollout.states[rollout.first_violation_step]
                rollout_axis.scatter(
                    violation[0],
                    violation[1],
                    color=linear_color,
                    marker="x",
                    s=42,
                    linewidths=1.5,
                    zorder=5,
                )
        rollout_points = np.concatenate(trajectory_arrays, axis=0)
        boundary_velocity = np.linspace(
            min(rollout_points[:, 1].min(), v_lo),
            min(max(rollout_points[:, 1].max(), v_hi), 0.0),
            500,
        )
        boundary_position = (
            np.maximum(-boundary_velocity, 0.0) ** 2
            / (2.0 * braking_acceleration)
            + contract_margin
        )
        rollout_axis.plot(
            boundary_position,
            boundary_velocity,
            color="#111111",
            linestyle=":",
            linewidth=1.4,
            zorder=3,
        )
        rollout_axis.plot(
            box[:, 0],
            box[:, 1],
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        set_local_limits(rollout_axis, [box, *trajectory_arrays])
        rollout_axis.set_title(f"Deterministic rollouts (up to {rollout_steps} steps)")

    for axis in axes:
        axis.set_xlabel("Relative position")
        axis.set_ylabel("Relative velocity")
        axis.grid(color="#d7dde1", linewidth=0.6)
        axis.set_aspect("equal", adjustable="box")

    legend_items = [
        Patch(facecolor=input_color, edgecolor="none", label="PWL input-safe domain"),
        Line2D([0], [0], color=region_color, linewidth=1.0, label="Affine region"),
    ]
    if evaluate_nonlinear:
        legend_items.append(
            Patch(
                facecolor=nonlinear_color,
                alpha=0.35,
                edgecolor="none",
                label="Region contains nonlinear violation",
            )
        )
    if evaluate_linear:
        legend_items.append(
            Patch(
                facecolor=linear_color,
                alpha=0.72,
                edgecolor="#7d241d",
                label="Exact linear counterexample subset",
            )
        )
    legend_items.extend(
        [
            Line2D([0], [0], color=safe_color, linewidth=1.3, label="PWL boundary"),
            Line2D(
                [0],
                [0],
                color="#111111",
                linestyle=":",
                linewidth=1.4,
                label="Nonlinear boundary",
            ),
        ]
    )
    if rollouts:
        legend_items.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="#245f9e",
                    marker="o",
                    linewidth=1.0,
                    label="Starts on nonlinear boundary",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#7b4f9d",
                    marker="^",
                    linewidth=1.0,
                    label="Starts inside boundary",
                ),
                Line2D(
                    [0],
                    [0],
                    color=linear_color,
                    marker="x",
                    linewidth=0,
                    label="First rollout violation",
                ),
            ]
        )
    figure.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_results(
    path: Path,
    regions: Sequence[AffineRegion],
    pieces: Sequence[CounterexamplePiece],
    rollouts: Sequence[Rollout],
) -> None:
    region_vertices, region_offsets = flatten_arrays(
        [region.vertices for region in regions]
    )
    region_post_vertices, region_post_offsets = flatten_arrays(
        [region.post_vertices for region in regions]
    )
    piece_vertices, piece_offsets = flatten_arrays(
        [piece.vertices for piece in pieces]
    )
    piece_post_vertices, piece_post_offsets = flatten_arrays(
        [piece.post_vertices for piece in pieces]
    )
    rollout_states, rollout_state_offsets = flatten_arrays(
        [rollout.states for rollout in rollouts]
    )
    rollout_actions, rollout_action_offsets = flatten_vectors(
        [rollout.actions for rollout in rollouts]
    )
    rollout_margins, rollout_margin_offsets = flatten_vectors(
        [rollout.margins for rollout in rollouts]
    )
    if regions:
        patterns = np.stack([region.pattern for region in regions])
        action_C = np.stack([region.action_C for region in regions])
        post_matrix = np.stack([region.post_matrix for region in regions])
        post_offset = np.stack([region.post_offset for region in regions])
    else:
        patterns = np.empty((0, 0), dtype=bool)
        action_C = np.empty((0, 2), dtype=np.float64)
        post_matrix = np.empty((0, 2, 2), dtype=np.float64)
        post_offset = np.empty((0, 2), dtype=np.float64)
    np.savez_compressed(
        path,
        region_vertices=region_vertices,
        region_vertex_offsets=region_offsets,
        region_post_vertices=region_post_vertices,
        region_post_vertex_offsets=region_post_offsets,
        activation_pattern=patterns,
        activation_pattern_digest=np.asarray(
            [pattern_digest(region.pattern) for region in regions]
        ),
        action_C=action_C,
        action_d=np.asarray([region.action_d for region in regions]),
        post_matrix=post_matrix,
        post_offset=post_offset,
        region_area=np.asarray([polygon_area(region.vertices) for region in regions]),
        linear_unsafe_region=np.asarray(
            [region.linear_unsafe for region in regions], dtype=bool
        ),
        nonlinear_unsafe_region=np.asarray(
            [region.nonlinear_unsafe for region in regions], dtype=bool
        ),
        nonlinear_minimum_margin=np.asarray(
            [region.nonlinear_minimum_margin for region in regions]
        ),
        linear_piece_vertices=piece_vertices,
        linear_piece_vertex_offsets=piece_offsets,
        linear_piece_post_vertices=piece_post_vertices,
        linear_piece_post_vertex_offsets=piece_post_offsets,
        linear_piece_affine_region_index=np.asarray(
            [piece.affine_region_index for piece in pieces], dtype=np.int64
        ),
        linear_piece_violated_constraint_index=np.asarray(
            [piece.violated_constraint_index for piece in pieces], dtype=np.int64
        ),
        linear_piece_area=np.asarray(
            [polygon_area(piece.vertices) for piece in pieces]
        ),
        rollout_start_kind=np.asarray(
            [rollout.start_kind for rollout in rollouts]
        ),
        rollout_states=rollout_states,
        rollout_state_offsets=rollout_state_offsets,
        rollout_actions=rollout_actions,
        rollout_action_offsets=rollout_action_offsets,
        rollout_margins=rollout_margins,
        rollout_margin_offsets=rollout_margin_offsets,
        rollout_first_violation_step=np.asarray(
            [rollout.first_violation_step for rollout in rollouts], dtype=np.int64
        ),
        rollout_first_violation_reason=np.asarray(
            [rollout.first_violation_reason for rollout in rollouts]
        ),
        rollout_termination_reason=np.asarray(
            [rollout.termination_reason for rollout in rollouts]
        ),
    )


def run(args: argparse.Namespace) -> dict:
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if args.action_scale <= 0.0 or args.braking_acceleration <= 0.0:
        raise ValueError("Acceleration values must be positive")
    if args.max_local_regions <= 0 or args.max_split_depth < 0:
        raise ValueError("Invalid VeriTex region or split limit")
    if args.rollout_steps <= 0 or args.rollouts_per_kind <= 0:
        raise ValueError("Rollout steps and counts must be positive")
    if args.rollout_inside_offset <= 0.0 or args.rollout_tolerance < 0.0:
        raise ValueError("Invalid rollout offset or tolerance")
    p_lo, p_hi, v_lo, v_hi = args.box
    if p_lo >= p_hi or v_lo >= v_hi:
        raise ValueError("--box bounds must be strictly increasing")

    checkpoint = resolve_checkpoint(args.model)
    model, model_metadata = load_policy(checkpoint, args.action_scale)
    network = FFNN(build_veritex_network(model))

    base_grid = load_safe_grid(args.safe_grid)
    post_safe_A = np.asarray(base_grid["safe_polytope_A"], dtype=np.float64)
    post_safe_b = np.asarray(base_grid["safe_polytope_b"], dtype=np.float64)
    input_grid = restrict_safe_grid(base_grid, args.box, args.tolerance)
    input_polygon = np.asarray(
        input_grid["safe_polygon_vertices"], dtype=np.float64
    )
    input_lower = input_polygon.min(axis=0)
    input_upper = input_polygon.max(axis=0)

    patterns, explored_nodes, subdivisions = discover_box_patterns(
        network,
        model,
        input_lower,
        input_upper,
        args.max_local_regions,
        args.max_split_depth,
        input_grid["safe_polytope_A"],
        input_grid["safe_polytope_b"],
        args.tolerance,
    )

    regions = []
    #print(patterns)
    #exit()
    for key in sorted(patterns):
        region = reconstruct_region(
            model, key, input_polygon, args.dt, args.tolerance
        )
        if region is not None:
            regions.append(region)
    regions.sort(key=lambda region: pattern_digest(region.pattern))

    input_area = polygon_area(input_polygon)
    covered_area = float(sum(polygon_area(region.vertices) for region in regions))
    coverage_error = abs(covered_area - input_area)
    coverage_tolerance = max(args.tolerance * 100.0, 1e-8) * max(input_area, 1.0)
    if coverage_error > coverage_tolerance:
        raise RuntimeError(
            "Affine regions do not cover the complete input domain: "
            f"domain area={input_area}, region area={covered_area}, "
            f"error={coverage_error}"
        )

    pieces = []
    evaluate_linear = args.property in ("linearized", "both")
    evaluate_nonlinear = args.property in ("nonlinear", "both")
    for region_index, region in enumerate(regions):
        if evaluate_linear:
            region_pieces = extract_linear_counterexamples(
                region_index,
                region,
                post_safe_A,
                post_safe_b,
                args.tolerance,
            )
            region.linear_unsafe = bool(region_pieces)
            pieces.extend(region_pieces)
        if evaluate_nonlinear:
            minimum_margin = nonlinear_minimum_margin(
                region.post_vertices, args.braking_acceleration
            )
            region.nonlinear_minimum_margin = minimum_margin
            region.nonlinear_unsafe = (
                minimum_margin < args.contract_margin - args.tolerance
            )

    rollouts = []
    if not args.no_rollouts:
        starts = make_rollout_starts(
            args.box,
            args.braking_acceleration,
            args.contract_margin,
            args.rollout_inside_offset,
            args.rollouts_per_kind,
        )
        maximum_position = float(base_grid["position_edges"][-1])
        rollouts = [
            simulate_rollout(
                model,
                start_kind,
                initial_state,
                args.rollout_steps,
                args.dt,
                args.action_scale,
                args.braking_acceleration,
                args.contract_margin,
                maximum_position,
                args.rollout_tolerance,
            )
            for start_kind, initial_state in starts
        ]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "counterexample_regions.npz"
    save_results(archive_path, regions, pieces, rollouts)
    plot_path = output_dir / "counterexample_regions.png"
    if not args.no_plot:
        plot_results(
            plot_path,
            args.box,
            input_polygon,
            base_grid["safe_polygon_vertices"],
            regions,
            pieces,
            evaluate_linear,
            evaluate_nonlinear,
            args.braking_acceleration,
            args.contract_margin,
            rollouts,
            args.rollout_steps,
        )

    linear_region_count = sum(region.linear_unsafe for region in regions)
    nonlinear_region_count = sum(region.nonlinear_unsafe for region in regions)
    summary = {
        "format": "acc_box_counterexample_regions_v1",
        "complete": True,
        "checkpoint": str(checkpoint),
        "safe_grid": str(args.safe_grid.expanduser().resolve()),
        "network_sizes": model_metadata["sizes"],
        "query_box": list(args.box),
        "input_domain_semantics": (
            "Query box intersected with the PWL ACC input-safe polytope."
        ),
        "property": args.property,
        "dt": args.dt,
        "action_scale": args.action_scale,
        "braking_acceleration": args.braking_acceleration,
        "contract_margin": args.contract_margin,
        "veritex_nodes_explored": explored_nodes,
        "adaptive_subdivisions": subdivisions,
        "activation_patterns": len(patterns),
        "nonempty_affine_regions": len(regions),
        "input_domain_area": input_area,
        "affine_region_area": covered_area,
        "affine_region_coverage_error": coverage_error,
        "linearized_unsafe_affine_regions": linear_region_count,
        "linearized_counterexample_polygons": len(pieces),
        "linearized_counterexample_area": float(
            sum(polygon_area(piece.vertices) for piece in pieces)
        ),
        "nonlinear_unsafe_affine_regions": nonlinear_region_count,
        "nonlinear_detection": (
            "Exact existence test: the braking margin is concave on each "
            "convex post-image, so its minimum is attained at a vertex."
        ),
        "nonlinear_subset_representation": (
            "Each flagged source polygon and its affine post_matrix/post_offset "
            "are stored; the curved unsafe subset is not polygonized."
        ),
        "rollout_steps": args.rollout_steps if rollouts else 0,
        "rollout_tolerance": args.rollout_tolerance if rollouts else None,
        "rollouts": [
            {
                "start_kind": rollout.start_kind,
                "initial_state": rollout.states[0].tolist(),
                "final_state": rollout.states[-1].tolist(),
                "completed_steps": len(rollout.actions),
                "minimum_margin": float(rollout.margins.min()),
                "minimum_position": float(rollout.states[:, 0].min()),
                "maximum_position": float(rollout.states[:, 0].max()),
                "minimum_action": float(rollout.actions.min()),
                "maximum_action": float(rollout.actions.max()),
                "first_violation_step": (
                    rollout.first_violation_step
                    if rollout.first_violation_step >= 0
                    else None
                ),
                "first_violation_reason": (
                    rollout.first_violation_reason or None
                ),
                "termination_reason": rollout.termination_reason,
            }
            for rollout in rollouts
        ],
        "plot": str(plot_path) if not args.no_plot else None,
    }
    summary_path = output_dir / "counterexample_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved regions: {archive_path}")
    print(f"Saved summary: {summary_path}")
    if not args.no_plot:
        print(f"Saved plot: {plot_path}")
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
