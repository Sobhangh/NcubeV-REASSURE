import argparse
import math
import importlib.util
import copy
from pathlib import Path
from time import time

import polytope as pc
import pickle
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

import onnx
from onnx2torch import convert
import torch

from REASSURE.REASSURE.Repair import REASSURERepair
#from REASSURE.ICLR.tools.build_PNN import MultiPointsPNN
import NCubeV.experiments.acc.training.acc as acc
import gymnasium as gym
from gymnasium.utils import seeding

#Is it possible to retrain a REASSURE patched network?
#Here is an error which says non linear operations may not be supported:
#  File "/home/ubuntu/NcubeV-REASSURE/NCubeV/src/Verifiers/../../deps/nnenum/src/nnenum/onnx_network.py", line 760, in load_onnx_network
#     assert o in Settings.ONNX_WHITELIST, f"Onnx model contains node with op {o}, which may not be a linear operation. " + \


def _is_gym_env_registered(env_id):
    from gymnasium.envs.registration import registry as gymnasium_registry
    try:
        return env_id in gymnasium_registry
    except Exception:
        return hasattr(gymnasium.envs, "registry") and env_id in getattr(gymnasium.envs.registry, "env_specs", {})


def ensure_acc_env_registered(env_id="acc-variant-v1"):
    if _is_gym_env_registered(env_id):
        return

    # Load the module that calls gym.register(...) as a side effect.
    acc_module_path = (
        Path(__file__).resolve().parent
        / "NCubeV"
        / "experiments"
        / "acc"
        / "training"
        / "acc.py"
    )
    if not acc_module_path.exists():
        raise FileNotFoundError(f"Could not find ACC env module at: {acc_module_path}")

    spec = importlib.util.spec_from_file_location("ncube_acc_env_register", str(acc_module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for: {acc_module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not _is_gym_env_registered(env_id):
        raise RuntimeError(f"Gym environment '{env_id}' is still not registered after importing {acc_module_path}")


class SB3GymCompatWrapper(gym.Wrapper):
    """Bridge legacy Gym envs using _reset/_step to the API expected by SB3 wrappers."""
    nb_step = 0
    MAX_STEPS = 410
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                if hasattr(self.env, "seed"):
                    self.env.seed(seed)
                elif hasattr(self.env, "_seed"):
                    self.env._seed(seed)
            except Exception:
                pass

        out = None
        self.nb_step = 0
        try:
            out = self.env.reset(seed=seed, options=options)
        except TypeError:
            try:
                out = self.env.reset()
            except Exception:
                out = None
        except NotImplementedError:
            out = None

        if out is None and hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "_reset"):
            out = self.env.unwrapped._reset()

        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        out = None
        try:
            out = self.env.step(action)
        except NotImplementedError:
            out = None
        except TypeError:
            out = None

        if out is None and hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "_step"):
            out = self.env.unwrapped._step(action)

        if isinstance(out, tuple) and len(out) == 5:
            return out

        obs, reward, done, info = out
        #print(f"{self.nb_step}: {out}")
        truncated = False
        terminated = bool(done and not truncated)
        self.nb_step += 1
        if self.nb_step >= self.MAX_STEPS:
            truncated = True
            terminated = False
            done = True
        return obs, reward, terminated, truncated, info


def evaluate_policy2(model, env, n_eval_episodes=100):
    episode_rewards = []
    nb_crashes = 0
    for i in range(n_eval_episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out

        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                action_tensor = model(torch.tensor([obs], dtype=torch.float32))
            action = np.array(action_tensor.cpu().numpy()).reshape(-1)

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                if terminated and info.get("crash", False):
                    nb_crashes += 1
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out
                done = bool(done)
                if done and info.get("crash", False):
                    nb_crashes += 1

            ep_reward += float(reward)
        #print(f"Episode reward: {ep_reward}, info: {info}, obs: {obs}, crash count: {nb_crashes}")
        episode_rewards.append(ep_reward)
        # if i % (n_eval_episodes // 5) == 0:
        #     print(f"total crashes {nb_crashes}")

    return float(np.mean(episode_rewards)), float(np.std(episode_rewards)), nb_crashes

def export_repaired_model_to_onnx(model, onnx_path, input_dim=2, opset_version=9):
    """Export repaired NNSum model to ONNX using a batch-shaped dummy input."""
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).eval()

    # Freeze all registered parameters.
    for param in export_model.parameters():
        param.requires_grad_(False)

    # MultiPNN stores branch modules/tensors in plain Python lists.
    # Freeze/detach them explicitly so exporter doesn't treat grad-enabled tensors as constants.
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

    onnx_model = OnnxableActionPolicy(export_model)
    dummy_input = torch.zeros((1, input_dim), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            onnx_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

def any_point_in_polytope(poly):
    # Option 1: center of largest inscribed ball (strictly inside if radius > 0)
    radius, center_raw = pc.cheby_ball(poly)
    center_arr = np.asarray(center_raw)
    center = np.array(
        [v for v in center_arr.reshape(-1)],
        dtype=np.float32,
    )
    # if center_arr.dtype == np.object_:
    #     # Flatten nested scalar containers into plain floats.
    #     center = np.array(
    #         [v for v in center_arr.reshape(-1)],
    #         dtype=np.float32,
    #     )
    # else:
    #     center = center_arr.astype(np.float32, copy=False).reshape(-1)

    #radius = float(np.asarray(radius).squeeze())
    center = torch.tensor(center, dtype=torch.float32)
    #print("Center:", center)
    if radius >= 0 and not torch.isnan(center).any():
        return center, radius  # radius > 0 means strictly interior
    #raise RuntimeError("Polytope appears infeasible.")

def plot_polytope(poly):
    try:
        # Some polytope versions expose plot directly.
        pc.plot(poly)
    except AttributeError:
        # Fallback: plot polygon boundary from extreme points.
        verts = np.asarray(pc.extreme(poly))
        if verts.ndim == 2 and verts.shape[0] > 0:
            center = verts.mean(axis=0)
            angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
            order = np.argsort(angles)
            polyline = verts[order]
            polyline = np.vstack([polyline, polyline[0]])
            plt.plot(polyline[:, 0], polyline[:, 1], "-", linewidth=2)
            plt.fill(polyline[:, 0], polyline[:, 1], alpha=0.25)
        else:
            raise ValueError("Could not extract vertices from the polytope.")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


class OnnxableActionPolicy(torch.nn.Module):
    def __init__(self, network):
        super(OnnxableActionPolicy, self).__init__()
        self.network = network
        normalize_linear1 = torch.nn.Linear(1, 4)
        # 100* ((max(0,x) - max(0,-x)) - max(0,x-1) + max(0,-x-1))
        normalize_linear1.weight.data = torch.Tensor([[1],[-1],[1],[-1]])
        normalize_linear1.bias.data=torch.Tensor([0,0,-1,-1])
        #print(normalize_linear1.weight)
        #print(normalize_linear1.bias)
        A = 100-1e-6
        normalize_linear2 = torch.nn.Linear(3,1)
        normalize_linear2.weight.data = torch.Tensor([[A,-A,-A,A]])
        normalize_linear2.bias.data=torch.Tensor([0])
        self.normalizer = torch.nn.Sequential(
            normalize_linear1,
            torch.nn.ReLU(),
            normalize_linear2)

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action = self.network(observation)
        return self.normalizer(action) #, self.value_net(value_hidden)

MINIMUM = 1
MAXIMUM = 1.2
def success_rate(model, buggy_inputs, is_print=0):
    with torch.no_grad():
        pred = model(buggy_inputs)
        #print("pred", pred)
        correct = ((pred >= MINIMUM) & (pred <= MAXIMUM)).type(torch.float).sum().item()
    if is_print == 1:
        print('Original accuracy on buggy_inputs: {} %'.format(100*correct / len(buggy_inputs)))
    elif is_print == 2:
        print('Success repair rate: {} %'.format(100*correct / len(buggy_inputs)))
    return correct/len(buggy_inputs)

def simulate_run(n=3000):
    rPos = [np.random.uniform(0, 100)]
    rVel = [-math.sqrt(rPos[0]*2*100)]
    state = [rPos[0], rVel[0]]
    print(f"Starting with: {state}")
    actions = []
    for i in range(0,n):
        action = repaired_model(torch.tensor([state], dtype=torch.float32)).item()
        action = max(-1, min(1, action)) * 100  # Clip action to [-100, 100]
        #action=100
        actions.append(action)
        t=0.1
        pos_0 = state[0]
        vel_0 = state[1]
        vel = action*t + vel_0
        pos = action*t**2/2 + vel_0*t + pos_0
        state[0]=pos
        state[1]=vel
        rPos.append(state[0])
        rVel.append(state[1])
        if state[0] <= 1 or state[0] > 100:
            if state[0] <= 0:
                print("CRASH")
            break
    return rPos, rVel, actions

#ensure_acc_env_registered('acc-variant-v1')
eval_env = gym.make('acc-variant-v1')
torch.manual_seed(42)
eval_env.np_random, _ = seeding.np_random(42)


parser = argparse.ArgumentParser(description="Retrain the ACC PPO policy using a polytope file and model base name.")
# parser.add_argument("POLYTOPE_FILE", help="Path to the polytope pickle file.")
# parser.add_argument("MODLE_FILE", help="Base filename for the PPO model, without the .zip extension.")
parser.add_argument("RUN_NB", type=int, help="Run number for naming outputs.")
parser.add_argument("UPPER_BOUND", type=float, help="Upper bound for the position constraint.")
args = parser.parse_args()
RUN_NB = args.RUN_NB
UPPER_BOUND = args.UPPER_BOUND

def format_upper_bound(value):
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, "g")

UPPER_BOUND_TAG = format_upper_bound(UPPER_BOUND)

POLYTOPE_FILE = ""
MODEL_FILE= ""
SMALL_MODEL = False
if SMALL_MODEL:
    POLYTOPE_FILE = "polytopes-small-approx-1.pkl"
    onnx_model = onnx.load("ppo_acc_small_200000_steps.onnx")
    torch_model = convert(onnx_model)
    # Keep only the policy feature extractor and the first hidden layer.
    # The ONNX graph is:
    # extractor/policy_net/policy_net/0/Gemm -> Relu ->
    # extractor/policy_net/policy_net/2/Gemm -> Relu -> action_net -> normalizer
    torch_model = torch.nn.Sequential(
        getattr(torch_model, "extractor/policy_net/policy_net/0/Gemm"),
        getattr(torch_model, "extractor/policy_net/policy_net/1/Relu"),
        getattr(torch_model, "extractor/policy_net/policy_net/2/Gemm"),
        getattr(torch_model, "extractor/policy_net/policy_net/3/Relu"),
        getattr(torch_model, "action_net/Gemm"),
    )
else:
    POLYTOPE_FILE = "path_RSSR/acc_bigger_polytopes"
    MODEL_FILE= "path_RSSR/ppo_acc_bigger_200000_steps"
    if RUN_NB > 1:
        poly_file = f"{POLYTOPE_FILE}-{RUN_NB-1}.pkl"
        model_file = f"{MODEL_FILE}-{UPPER_BOUND_TAG}-{RUN_NB-1}.pt"
    else:
        poly_file = POLYTOPE_FILE + ".pkl"
        model_file = MODEL_FILE + ".pt"
    # onnx_model = onnx.load(model_file)
    # print("\nONNX initializers (stored parameters):")
    # onnx_total_params = 0
    # for init in onnx_model.graph.initializer:
    #     arr = onnx.numpy_helper.to_array(init)
    #     n = int(arr.size)
    #     onnx_total_params += n
    #     print(f"{init.name:40s} shape={tuple(arr.shape)} dtype={arr.dtype} numel={n}")

    # print(f"Total ONNX initializer params: {onnx_total_params}")
    # 
    torch_model = torch.load(model_file, weights_only=False)
print(f"Using model file: {model_file}, polytope file: {poly_file}")
#y = torch_model(torch.tensor([[0.0, 0.0]], dtype=torch.float32))
# for name, param in torch_model.named_parameters():
#     print(name, param.shape)
total_params = sum(p.numel() for p in torch_model.parameters())
print(f"Total parameters in original model: {total_params}")

# mean_reward, std_reward, crashes = evaluate_policy2(torch_model, eval_env, n_eval_episodes=1000)
# print(f"********* Original model - mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}, crashes: {crashes} in 1000 episodes")
#One of the runs: ********* Original model - mean_reward:2451.84 +/- 2658.49, crashes: 13
# With the polytope from the repo: ********* Original model - mean_reward:2926.55 +/- 2333.95, crashes: 9
# With 1000 episodes: ********* Original model - mean_reward:3086.22 +/- 2188.86, crashes: 152


# Read the polytope.pkl file
#"acc-2000000-64-64-64-64-polytopes.pickle"
#polytopes-approx-3.pkl
#It seems like the one with bound produced by the script is most reliable given that
#it has no infeasible patches. 
#acc-2000000-64-64-64-64-polytopes.pickle
#polytopes-approx-3-bound.pkl
with open(poly_file, 'rb') as f:
    polytopes = pickle.load(f)
#A, b = polytopes[0].A, polytopes[0].b
print("Number of polytopes:", len(polytopes))

input_dim = 2
input_boundary = [np.array([-0, -200]), np.array([100, 200])]
A = np.array([[1.0, 0.0],
              [-1.0, 0.0]])
print("Starting the repair process...\n")
set_upper_bound_list = [UPPER_BOUND]  #10, 20, 30, 100
for set_upper_bound in set_upper_bound_list:
    print(f"set_upper_bound: {set_upper_bound}")
    b = np.array([set_upper_bound, 0.0])
    position_bound_poly = pc.Polytope(A, b)

    intersected_poly = []
    for poly in polytopes:
        # isect = poly.intersect(position_bound_poly)
        # if not pc.is_empty(isect):
        #     intersected_poly.append(isect)
        intersected_poly.append(poly)
    print("Intersected non-empty polytopes:", len(intersected_poly))


    buggy_inputs = torch.stack(list(map(lambda p: any_point_in_polytope(p)[0], list(filter(lambda p: any_point_in_polytope(p) is not None, intersected_poly)))))
    
    # input_boundary = [np.block([[np.eye(input_dim)], [-np.eye(input_dim)]]),
    #                       np.block([np.array([100, 200]), np.array([0, 200])])]
    output_constraints = ([np.array([[-1], [1]])] * len(intersected_poly),[np.array([-MINIMUM, MAXIMUM])] * len(intersected_poly))

    success_rate(torch_model, buggy_inputs, is_print=1)
    start = time()


    REASSURE = REASSURERepair(torch_model, input_boundary, n=1)
    repaired_model = REASSURE.polytope_wise_repair(intersected_poly, output_constraints=output_constraints, core_num=1)

    # n = 10
    # input_boundary = [np.array([200, 200]), np.array([-200, -200])]
    # PNN = MultiPointsPNN(torch_model, n, input_boundary, test_model=False)
    # PNN.area_repair(buggy_linear_regions, P, ql, qu)
    # repaired_model = PNN.compute(4)

    cost_time = time()-start
    print('Patching Time of REASSURE:', cost_time)
    success_rate(repaired_model, buggy_inputs, is_print=2)

    # for name, param in repaired_model.named_parameters():
    #     print(name, param.shape)
    # for layer in repaired_model.pnn.layer_list:
    #     for name, param in layer.named_parameters():
    #         print(name, param.shape)
    #print(f"layer list length: {len(repaired_model.pnn.layer_list)}")
    
    full_model_path = f"{MODEL_FILE}-{UPPER_BOUND_TAG}-{RUN_NB}.pt"
    onnx_model_path = f"{MODEL_FILE}-{UPPER_BOUND_TAG}-{RUN_NB}.onnx"
    Path(full_model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(repaired_model, full_model_path)
    print(f"Saved full model to: {full_model_path}")
    try:
        export_repaired_model_to_onnx(repaired_model, onnx_model_path, input_dim=input_dim)
        if not Path(onnx_model_path).is_file():
            raise FileNotFoundError(f"ONNX export reported success but file was not created: {onnx_model_path}")
        print(f"Saved ONNX model to: {onnx_model_path}")
    except Exception as e:
        raise RuntimeError(
            f"ONNX export failed for upper bound {set_upper_bound}: {type(e).__name__}: {e}"
        ) from e
        
    total_params = sum(p.numel() for p in repaired_model.target_nn.parameters())
    additional_params = sum(sum(p.numel() for p in layer.parameters()) for layer in repaired_model.pnn.layer_list)
    additional_params += sum(sum(p.numel() for p in g.parameters()) for g in repaired_model.pnn.g_list)
    additional_params += len(repaired_model.pnn.K_list)  # Each K is a scalar parameter
    total_params += additional_params
    print(f"Total parameters in repaired model: {total_params}")
    print(f"Additional parameters introduced by repair: {additional_params}")


    # for i in range(0):
    #     rPos, rVel, actions = simulate_run(n=4000)
        
    # mean_reward, std_reward, crashes_repaired = evaluate_policy2(repaired_model, eval_env, n_eval_episodes=20)
    # print(f"********* Repaired model - mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}, crashes: {crashes_repaired}")
    #One run: ********* Repaired model - mean_reward:3866.36 +/- 847.17, crashes_repaired: 1

    #bigger network
    # With the polytope from the repo: ********* Repaired model - mean_reward:2446.47 +/- 2420.59, crashes_repaired: 3
    # with 1000 episodes: ********* Repaired model - mean_reward:1838.93 +/- 2660.60, crashes_repaired: 125

