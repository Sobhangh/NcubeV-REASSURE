from pathlib import Path
import copy
from stable_baselines3 import PPO

import numpy as np
import onnxruntime as ort
import torch

# Ensure class definitions are importable for pickle-based full-object loading.
from REASSURE.ICLR.tools.build_PNN import NNSum  # noqa: F401


CHECKPOINT_PATH = Path("acc_repaired_model_upper_bound_10.pt")
ONNX_PATH = Path("acc_repaired_model_upper_bound_10.onnx")
NUM_TEST_INPUTS = 512
INPUT_LOW = np.array([0.0, -200.0], dtype=np.float32)
INPUT_HIGH = np.array([100.0, 200.0], dtype=np.float32)


def load_torch_model(checkpoint_path: Path):
	# PyTorch 2.6+ defaults to weights_only=True, which rejects custom classes.
	loaded_obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
	if isinstance(loaded_obj, dict):
		raise TypeError(
			"Loaded object is a state_dict, not an NNSum model. "
			"Rebuild the model architecture first, then call load_state_dict(...)."
		)
	loaded_obj.eval()
	return loaded_obj


def export_to_onnx_if_missing(model, onnx_path: Path):
	if onnx_path.exists():
		return
	export_model = copy.deepcopy(model).eval()
	for param in export_model.parameters():
		param.requires_grad_(False)

	# MultiPNN stores branches in plain Python lists; freeze them explicitly.
	if hasattr(export_model, "pnn"):
		if hasattr(export_model.pnn, "g_list"):
			for g in export_model.pnn.g_list:
				if hasattr(g, "parameters"):
					for param in g.parameters():
						param.requires_grad_(False)
		if hasattr(export_model.pnn, "layer_list"):
			for layer in export_model.pnn.layer_list:
				if hasattr(layer, "parameters"):
					for param in layer.parameters():
						param.requires_grad_(False)
		if hasattr(export_model.pnn, "K_list"):
			export_model.pnn.K_list = [
				k.detach().clone() if torch.is_tensor(k) else torch.tensor(float(k), dtype=torch.float32)
				for k in export_model.pnn.K_list
			]

	dummy_input = torch.zeros((1, 2), dtype=torch.float32)
	with torch.no_grad():
		torch.onnx.export(
			export_model,
			dummy_input,
			str(onnx_path),
			export_params=True,
			opset_version=18,
			dynamo=False,
			do_constant_folding=True,
			input_names=["input"],
			output_names=["output"],
			dynamic_axes={
				"input": {0: "batch_size"},
				"output": {0: "batch_size"},
			},
		)


def make_inputs(num_inputs: int):
	rng = np.random.default_rng(seed=42)
	samples = rng.uniform(low=INPUT_LOW, high=INPUT_HIGH, size=(num_inputs, 2)).astype(np.float32)
	return samples


def main():
	model = PPO.load(f"retrain/ppo_acc_bigger_200000_steps.zip")
	for name, param in model.policy.value_net.named_parameters():
		print(name, param.shape)
	print("extractor")
	for name, param in model.policy.mlp_extractor.named_parameters():
			print(name, param.shape)
	print("action_net")
	for name, param in model.policy.action_net.named_parameters():
		print(name, param.shape)
	print("all policy")
	for name, param in model.policy.named_parameters():
		print(name, param.shape)
	exit()

	repaired_model = load_torch_model(CHECKPOINT_PATH)
	export_to_onnx_if_missing(repaired_model, ONNX_PATH)

	sample = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
	with torch.no_grad():
		y = repaired_model(sample)
	print("Torch output at [0, 0]:", y)

	total_params = sum(p.numel() for p in repaired_model.target_nn.parameters())
	additional_params = sum(sum(p.numel() for p in layer.parameters()) for layer in repaired_model.pnn.layer_list)
	additional_params += sum(sum(p.numel() for p in g.parameters()) for g in repaired_model.pnn.g_list)
	additional_params += len(repaired_model.pnn.K_list)  # Each K is a scalar parameter
	total_params += additional_params
	print(f"Total parameters in repaired model: {total_params}")
	print(f"Additional parameters introduced by repair: {additional_params}")

	ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
	input_name = ort_session.get_inputs()[0].name

	x_np = make_inputs(NUM_TEST_INPUTS)
	x_torch = torch.from_numpy(x_np)

	with torch.no_grad():
		y_torch = repaired_model(x_torch).cpu().numpy()
	y_onnx = ort_session.run(None, {input_name: x_np})[0]

	abs_diff = np.abs(y_torch - y_onnx)
	max_abs_diff = float(abs_diff.max())
	mean_abs_diff = float(abs_diff.mean())
	p99_abs_diff = float(np.quantile(abs_diff, 0.99))

	print(f"Compared on {NUM_TEST_INPUTS} random inputs")
	print(f"ONNX file: {ONNX_PATH}")
	print(f"max|torch-onnx| = {max_abs_diff:.8e}")
	print(f"mean|torch-onnx| = {mean_abs_diff:.8e}")
	print(f"p99|torch-onnx| = {p99_abs_diff:.8e}")

	# Print the worst-case input pair for quick debugging.
	worst_idx = int(np.argmax(abs_diff.reshape(-1)))
	row_idx = worst_idx // abs_diff.shape[1]
	print("Worst-case input:", x_np[row_idx])
	print("Torch output:", y_torch[row_idx])
	print("ONNX output:", y_onnx[row_idx])


if __name__ == "__main__":
	main()