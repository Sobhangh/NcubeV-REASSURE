We use veritex to enumerate unsafe regions. These are then passed to REASSURE for repair.

## ACC safe grid

`compute_acc_safe_grid.py` reconstructs the WNN velocity thresholds and uses
them as breakpoints for a conservative piecewise-linear approximation of the
continuous braking invariant. Secants lie above the convex braking curve, so
the resulting safe polytope is an inner approximation of the true safe set.

```bash
python baselines/REASSURE/compute_acc_safe_grid.py \
  runs/acc-variant-v1__acc_dwc_b61_1024x2_n6__SEED__RUN
```

The output archive contains the safe polytope as `safe_polytope_A` and
`safe_polytope_b`, its polygon vertices, linear boundary pieces, and masks for
grid boxes that are inside, intersect, or lie outside the polytope. The union of
inside and boundary boxes is `discovery_cells`, which can drive complete local
VeriTex discovery; final ReLU regions should be intersected with the polytope,
not with the grid boxes. No dynamics or fixed-point iteration is performed.

## One-step unsafe ReLU regions

`find_acc_unsafe_regions.py` uses the safe-grid boxes only to discover every
ReLU activation pattern that could be unsafe. It reconstructs each candidate
as a global polytope, intersects it directly with the PWL safe set, and
propagates the result through one discrete ACC step. The final action clamp is
included in the activation pattern, so the physical action is affine on every
reported region.

Cheap interval bounds first certify cells and post-image boxes that are far
from the safety boundary. All remaining regions are classified exactly by
their affine post-image vertices against the continuous braking invariant.
For initial experiments, `--region P_LO P_HI V_LO V_HI` restricts both
discovery and the final polytopes to a small rectangular region of interest.

```bash
python baselines/REASSURE/find_acc_unsafe_regions.py \
  runs/acc-variant-v1__acc_relu_64x4__SEED__RUN \
  runs/reassure_safe_grid/CHECKPOINT/safe_grid_cells.npz \
  --output-dir runs/reassure_one_step/SEED \
  --region 10 30 -70 -40
```

The output consists of `one_step_summary.json`, a portable flattened archive
`one_step_affine_regions.npz`, and a source/post-image plot. A run is complete
only when the summary has `"complete": true`; `--max-discovery-cells` is a
smoke-test option and deliberately marks its output incomplete.
