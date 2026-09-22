"""Measure a REASSURE patch against the original ACC PPO on paired states."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.acc.make_acc_quality_table import Region
from training.evaluate_acc import SB3Actor
from REASSURE.ICLR.tools.build_PNN import NNSum  # required for torch.load


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="Original PPO .zip")
    parser.add_argument("--repaired", type=Path, required=True, help="REASSURE NNSum .pt")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--position-max", type=float, default=100.0)
    parser.add_argument("--region", choices=Region.MODES, default="closing-box")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        parser.error("--samples and --batch-size must be positive")

    torch.set_num_threads(1)
    original = SB3Actor(args.original).eval()
    repaired = torch.load(args.repaired, map_location="cpu", weights_only=False).eval()
    if not isinstance(repaired, NNSum):
        raise TypeError("REASSURE checkpoint must contain an NNSum controller")

    original_parameters = sum(parameter.numel() for parameter in original.parameters())
    patch_network_weights = sum(
        parameter.numel()
        for layer in (*repaired.pnn.g_list, *repaired.pnn.layer_list)
        for parameter in layer.parameters()
    )
    patch_scalar_constants = len(repaired.pnn.K_list)
    patch_parameters = patch_network_weights + patch_scalar_constants
    if original_parameters != sum(p.numel() for p in repaired.target_nn.parameters()):
        raise ValueError("original PPO and patched base network have different parameter counts")

    region = Region(args.region, args.position_max, 100.0, 200.0)
    # Draw the full sample before batching inference, exactly as the other
    # ACC delta tool does.  This keeps seed 0's states identical across runs.
    states = region.sample(args.samples, np.random.default_rng(args.seed))
    differences = np.empty(args.samples, dtype=np.float64)
    maximum_base_difference = 0.0
    with torch.inference_mode():
        for start in range(0, args.samples, args.batch_size):
            end = min(start + args.batch_size, args.samples)
            tensor = torch.as_tensor(states[start:end], dtype=torch.float32)
            old = original.get_action_and_value(tensor, deterministic=True)[0]
            base = repaired.target_nn(tensor).clamp(-1.0, 1.0)
            maximum_base_difference = max(
                maximum_base_difference, float((old - base).abs().max())
            )
            # SB3 and the exported ONNX controller both clip to the physical
            # action range.  Compare normalized actions, as in the ACC table.
            new = repaired(tensor).clamp(-1.0, 1.0)
            differences[start:end] = (
                (new - old).abs().amax(dim=1).double().cpu().numpy()
            )
    if maximum_base_difference > 1e-5:
        raise ValueError(
            f"original PPO differs from the patched base: {maximum_base_difference}"
        )

    result = {
        "format": "acc-sampled-controller-delta",
        "version": 1,
        "original_controller": str(args.original.resolve()),
        "repaired_controller": str(args.repaired.resolve()),
        "original_sha256": sha256(args.original),
        "repaired_sha256": sha256(args.repaired),
        "samples": args.samples,
        "seed": args.seed,
        "paired_samples": True,
        "region": {
            "mode": region.mode,
            "position_max": region.position_max,
            "braking_acceleration": region.braking_acceleration,
            "velocity_max": region.velocity_max,
            "description": region.describe(),
        },
        "action_dimensions": 1,
        "mean_abs_delta": float(differences.mean()),
        "standard_error": float(differences.std(ddof=1) / math.sqrt(args.samples))
        if args.samples > 1 else 0.0,
        "change_percent_exact_nonzero": 100.0 * float(np.mean(differences > 0)),
        "change_percent_by_abs_delta_threshold": {
            f"{threshold:g}": 100.0 * float(np.mean(differences > threshold))
            for threshold in (1e-6, 1e-4, 1e-3, 1e-2)
        },
        "original_parameters": original_parameters,
        "patch_parameters": patch_parameters,
        "patch_network_weights": patch_network_weights,
        "patch_scalar_constants": patch_scalar_constants,
        "repaired_parameters": original_parameters + patch_parameters,
        "parameter_increase_percent": 100.0 * patch_parameters / original_parameters,
        "parameter_count_convention": "patch weights and biases plus one K scalar per region",
        "patch_regions": len(repaired.pnn.g_list),
        "max_base_controller_difference": maximum_base_difference,
        "formal_status": "not verified by NCubeV",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
