import argparse

import torch
import numpy as np
import gymnasium as gym
import pickle
import polytope as pc
import copy
import onnx
from gymnasium.utils import seeding
import importlib.util
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import NCubeV.experiments.acc.training.acc as acc

# Set this to "cpu" or "cuda" at the top of the file.
# Example: DEVICE = "cpu"  or  DEVICE = "cuda"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OnnxableActionPolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super(OnnxableActionPolicy, self).__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net
        normalize_linear1 = torch.nn.Linear(1, 4)
        normalize_linear1.weight.data = torch.Tensor([[1], [-1], [1], [-1]])
        normalize_linear1.bias.data = torch.Tensor([0, 0, -1, -1])
        a_value = 100 - 1e-6
        normalize_linear2 = torch.nn.Linear(3, 1)
        normalize_linear2.weight.data = torch.Tensor([[a_value, -a_value, -a_value, a_value]])
        normalize_linear2.bias.data = torch.Tensor([0])
        self.normalizer = torch.nn.Sequential(
            normalize_linear1,
            torch.nn.ReLU(),
            normalize_linear2,
        )

    def forward(self, observation):
        action_hidden, value_hidden = self.extractor(observation)
        action = self.action_net(action_hidden)
        return self.normalizer(action)


def export_model_artifacts(model, zip_path, onnx_path, input_dim=2, opset_version=9):
    zip_path = Path(zip_path)
    onnx_path = Path(onnx_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(zip_path))

    onnxable_model = OnnxableActionPolicy(
        model.policy.mlp_extractor,
        model.policy.action_net,
        model.policy.value_net,
    )
    # onnxable_model.graph.output[0].name = "out1"
    # onnxable_model.graph.node[len(onnxable_model.graph.node)-1].output[0]="out1"

    dummy_input = torch.randn(1, input_dim)
    with torch.no_grad():
        torch.onnx.export(
            onnxable_model,
            dummy_input,
            str(onnx_path),
            opset_version=opset_version,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx_model.ir_version = min(onnx_model.ir_version, 7)
    onnx.save(onnx_model, str(onnx_path))

def alt_method():
    def cheby_ball(poly1):
        #logger.debug('cheby ball')
        if (poly1._chebXc is not None) and (poly1._chebR is not None):
            # In case chebyshev ball already calculated and stored
            return poly1._chebR, poly1._chebXc
        if isinstance(poly1, pc.Region):
            maxr = 0
            maxx = None
            for poly in poly1.list_poly:
                rc, xc = cheby_ball(poly)
                if rc > maxr:
                    maxr = rc
                    maxx = xc
            poly1._chebXc = maxx
            poly1._chebR = maxr
            return maxr, maxx
        if pc.is_empty(poly1):
            return 0, None
        # `poly1` is nonempty
        r = 0
        xc = None
        A = poly1.A
        c = np.negative(np.r_[np.zeros(np.shape(A)[1]), 1])
        norm2 = np.sqrt(np.sum(A * A, axis=1))
        G = np.c_[A, norm2]
        h = poly1.b
        sol = lpsolve(c, G, h)
        #return sol
        if sol['status'] == 0 or (sol['status'] == 4 and pc.is_inside(poly1,sol['x'][0:-1])):
            r = sol['x'][-1]
            if r < 0:
                return 0, None
            xc = sol['x'][0:-1]
        else:
            # Polytope is empty
            poly1 = pc.Polytope(fulldim=False)
            return 0, None
        poly1._chebXc = np.array(xc)
        poly1._chebR = np.double(r)
        return poly1._chebR, poly1._chebXc


    np_random, _ = seeding.np_random(42)
    POLYTOPE_VOLUMES = []
    POLYTOPES = []
    def init_polytopes(self,polytopes):
        volume = []
        for p in polytopes:
            volume.append(pc.volume(p))
        total_volume = sum(volume)
        
        POLYTOPE_VOLUMES = [0]
        for v in volume:
            POLYTOPE_VOLUMES.append((POLYTOPE_VOLUMES[-1]*total_volume + v)/total_volume)
        POLYTOPES = []  
        for p in polytopes:
            cheby_ball(p)
            POLYTOPES.append(p)

    def sample_from_poly():
        while True:
            #print(">", end="")
            r = np_random.uniform(low=0.0, high=1.0, size=(1,))[0]
            poly = POLYTOPES[-1]
            # TODO(steuber): Could be more efficient through binary search
            for i in range(len(POLYTOPE_VOLUMES)):
                if r > POLYTOPE_VOLUMES[i]:
                    poly = POLYTOPES[i-1]
            l_b, u_b = poly.bounding_box
            l_b = l_b.flatten()
            u_b = u_b.flatten()
            x = None
            n = poly.A.shape[1]
            for i in range(400):
                #print(".", end="")
                x = np_random.uniform(low=l_b,high=u_b,size=(n,))
                if x in poly:
                    break
            # Fallback if random sampling doesn't work
            if x is None:
                x = poly.chebXc
            # Fallback if polytope looks empty
            if x is None:
                continue
            return x

    A = 100
    B = 100
    MAX_VALUE = 200
    def is_crash(self, some_state):
        return some_state[0] <= 0

    def safe_polytope():
        while True:
            res = sample_from_poly()
            if -np.sqrt(res[0]*2*A)+1e-3<=res[1] and res[1]<=np.sqrt((MAX_VALUE-res[0])*2*B)-1e-3 and not (is_crash(None, res) or res[0] > MAX_VALUE):
                rv = res
                break
        return rv

def evaluate_policy2(model, env, n_eval_episodes=100):
    device = next(model.parameters()).device
    episode_rewards = []
    nb_crashes = 0
    for _ in range(n_eval_episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out

        done = False
        ep_reward = 0.0
        while not done:
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, _, _ = model(obs_tensor)
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

    return float(np.mean(episode_rewards)), float(np.std(episode_rewards)), nb_crashes


parser = argparse.ArgumentParser(description="Supervised retraining for the ACC PPO policy.")
parser.add_argument("RUN_NB", type=int, help="Run number for naming outputs.")
args = parser.parse_args()

RUN_NB = args.RUN_NB

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "supervised"

env = gym.make('acc-variant-v1')

env.unwrapped.INCLUDE_UNWINNABLE = False

env.np_random, seed = seeding.np_random(42)

POLYTOPE_FILE = ""
MODLE_FILE = ""
SMALL_MODLE = False
if SMALL_MODLE:
    POLYTOPE_FILE = "polytopes-small-approx-1.pkl"
    MODLE_FILE = "ppo_acc_small_200000_steps.zip"
else:
    POLYTOPE_FILE = SCRIPT_DIR / "supervised" / "acc-2000000-64-64-64-64-polytopes.pkl"
    if RUN_NB > 1:
        POLYTOPE_FILE = SCRIPT_DIR / "supervised" / f"acc-2000000-64-64-64-64-polytopes-{RUN_NB - 1}.pkl"
        MODLE_FILE = str(OUTPUT_DIR / f"ppo_acc_bigger_200000_steps-{RUN_NB - 1}.zip")
    else:
        MODLE_FILE = str(OUTPUT_DIR / "ppo_acc_bigger_200000_steps.zip")

retrain_polytopes = None
with open(POLYTOPE_FILE,"rb") as f:
    retrain_polytopes = pickle.load(f)

poly_region = pc.Region(retrain_polytopes)

env.unwrapped.init_polytopes(0.0,retrain_polytopes)

BUGGY_POINT_LEN = 100_000
buggy_points = []
for i in range(BUGGY_POINT_LEN):
    point, _ = env.unwrapped.polytope_reset()
    buggy_points.append(point)

MAX_STEPS = 410
def collect_obs_act(model, env, n_eval_episodes=BUGGY_POINT_LEN):
    device = next(model.parameters()).device
    obs_act_list = []
    while len(obs_act_list) < n_eval_episodes:
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out

        done = False
        crash = False
        ep_reward = 0.0
        episode_obs_act = []
        nb_step = 0
        
        while not done and nb_step < MAX_STEPS:
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, _, _ = model(obs_tensor)
            #print(f"obs: {obs}, action_tensor: {action_tensor}")
            action = np.array(action_tensor.cpu().numpy()).reshape(-1)

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                if terminated and info.get("crash", False):
                    crash = True
                    #print(info)
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out
                done = bool(done)
                if done and info.get("crash", False):
                    crash = True
            
            ep_reward += float(reward)
            episode_obs_act.append((obs, action))
            nb_step += 1
        if not crash:
            obs_act_list.extend(episode_obs_act)
    return obs_act_list


env2 = gym.make('acc-variant-v1')

env2.unwrapped.INCLUDE_UNWINNABLE = False

env2.np_random, seed = seeding.np_random(42)
torch.manual_seed(42)

model = PPO.load(MODLE_FILE, device=DEVICE)
model.policy.to(DEVICE)
model.set_env(env2)


print("Collecting observations and actions from the model...")
obs_act_list = collect_obs_act(model.policy, env2)
print(f"Collected {len(obs_act_list)} observations and actions from the model.")
# Supervised learning loop
import torch.nn as nn
import torch.optim as optim

# Prepare training data
train_obs = []
train_actions = []

# Data from buggy_points (correct label is 1)
for point in buggy_points:
    train_obs.append(torch.tensor(point, dtype=torch.float32, device=DEVICE))
    train_actions.append(torch.tensor([1.02], dtype=torch.float32, device=DEVICE))

# Data from obs_act_list (correct output is the action from the tuple)
for obs, action in obs_act_list:
    train_obs.append(torch.tensor(obs, dtype=torch.float32, device=DEVICE))
    train_actions.append(torch.tensor(action, dtype=torch.float32, device=DEVICE))

train_obs = torch.stack(train_obs)
train_actions = torch.stack(train_actions)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.policy.parameters(), lr=1e-5)
n_epochs = 20
batch_size = 256

mean_reward, std_reward, nb_crashes = evaluate_policy2(model.policy, env2, n_eval_episodes=1000)
print(f"Before retraining, mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}, crashes: {nb_crashes}")
#Before retraining, mean_reward: 2454.95 +/- 1691.29, crashes: 11

print(f"Starting supervised retraining for with batch size {batch_size}...")
# Training loop
#for epoch in range(n_epochs):
Loss = 100
epoch = 0
while Loss > 0.06:
    for i in range(0, len(train_obs), batch_size):
        batch_obs = train_obs[i:i+batch_size].to(DEVICE)
        batch_actions = train_actions[i:i+batch_size].to(DEVICE)
        
        optimizer.zero_grad()
        predictions, _, _ = model.policy(batch_obs)
        loss = criterion(predictions, batch_actions)
        loss.backward()
        optimizer.step()
        Loss = loss.item()
        #print(f"Epoch {epoch+1}/{n_epochs}, Batch {i//batch_size + 1}/{len(train_obs)//batch_size + 1}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    epoch += 1

final_zip_path = OUTPUT_DIR / f"ppo_acc_bigger_200000_steps-{RUN_NB}.zip"
final_onnx_path = OUTPUT_DIR / f"ppo_acc_bigger_200000_steps-{RUN_NB}.onnx"
export_model_artifacts(model, final_zip_path, final_onnx_path)
print(f"Saved PPO model to: {final_zip_path}")
print(f"Saved ONNX model to: {final_onnx_path}")

mean_reward, std_reward, nb_crashes = evaluate_policy2(model.policy, env2, n_eval_episodes=500)
print(f"After retraining, mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}, crashes: {nb_crashes}")

raise SystemExit(0 if nb_crashes == 0 else 1)

    

#After retraining, mean_reward: 2468.28 +/- 1768.67, crashes: 5