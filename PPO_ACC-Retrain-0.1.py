

# # Retraining after first attempt to prove correct


import argparse
import gymnasium as gym
import time

import pickle
import numpy as np
import polytope as pc

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from torch import nn

from gymnasium.utils import seeding
import NCubeV.experiments.acc.training.acc as acc

class OnnxableActionPolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super(OnnxableActionPolicy, self).__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net
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
        action_hidden, value_hidden = self.extractor(observation)
        action = self.action_net(action_hidden)
        return self.normalizer(action) #, self.value_net(value_hidden)

def evaluate_policy2(model, env, n_eval_episodes=100):
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
            with torch.no_grad():
                action_tensor,_,_ = model(torch.tensor([obs], dtype=torch.float32))
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
        #print(f"Episode reward: {ep_reward}, info: {info}, crash count: {nb_crashes}")
        episode_rewards.append(ep_reward)

    return float(np.mean(episode_rewards)), float(np.std(episode_rewards)), nb_crashes

parser = argparse.ArgumentParser(description="Retrain the ACC PPO policy using a polytope file and model base name.")
# parser.add_argument("POLYTOPE_FILE", help="Path to the polytope pickle file.")
# parser.add_argument("MODLE_FILE", help="Base filename for the PPO model, without the .zip extension.")
parser.add_argument("RUN_NB", type=int, help="Run number for naming outputs.")
args = parser.parse_args()

RUN_NB = args.RUN_NB
POLYTOPE_FILE = "retrain/acc_bigger_polytopes"
MODLE_FILE = "retrain/ppo_acc_bigger_200000_steps-RETRAIN"
SMALL_MODLE = False
print(f"Using model file: {MODLE_FILE} and polytope file: {POLYTOPE_FILE} (run {RUN_NB})")
    
torch.manual_seed(42)

env = gym.make('acc-variant-v1')

env.unwrapped.INCLUDE_UNWINNABLE = False

env.np_random, seed = seeding.np_random(42)

retrain_polytopes = None
if RUN_NB > 1:
    poly_file = f"{POLYTOPE_FILE}-{RUN_NB-1}.pkl"
else:
    poly_file = POLYTOPE_FILE + ".pkl"
with open(poly_file,"rb") as f:
    retrain_polytopes = pickle.load(f)

poly_region = pc.Region(retrain_polytopes)

torch.manual_seed(42)

eval_episode_length=500
training_episode_length=100_000

if RUN_NB > 1:
    model = PPO.load(f"{MODLE_FILE}-{RUN_NB -1}.zip")
else:
    model = PPO.load(f"retrain/ppo_acc_bigger_200000_steps.zip")
model.set_env(env)

# env.unwrapped.init_polytopes(1.0,[])
# mean_reward, std_reward, nb_crashes = evaluate_policy2(model.policy, env, n_eval_episodes=eval_episode_length)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}, crashes: {nb_crashes}")
# #mean_reward:2989.77 +/- 1977.46, crashes: 


# env.unwrapped.init_polytopes(0.0,retrain_polytopes)
# mean_reward, std_reward, nb_crashes = evaluate_policy2(model.policy, env, n_eval_episodes=eval_episode_length)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}, crashes: {nb_crashes}")
#mean_reward:590.70 +/- 2958.68, crashes: 

#Bigger model:
# mean_reward:2435.90 +/- 1708.01, crashes: 13
# mean_reward:348.00 +/- 2624.06, crashes: 553
# SMALL MODEL
#OVERALL: mean_reward:2961.51 +/- 1411.77, crashes: 1
#FOCUSED: mean_reward:-762.27 +/- 2307.03, crashes: 777


results_overall = {}
results_polys = {}

for p in [0.1]:
    # model = PPO.load(f"retrain/{MODLE_FILE}.zip")
    # model.set_env(env)
    
    #print("p=",p)

    env.unwrapped.init_polytopes(p,retrain_polytopes)
    env.unwrapped.INCLUDE_UNWINNABLE = False
    start_time = time.time()
    model=model.learn(total_timesteps=training_episode_length, log_interval=1000)
    print("--- %s seconds ---" % (time.time() - start_time))

    model.save(f"{MODLE_FILE}-{RUN_NB}.zip")

# #Performance of models on focus polytopes only?
# for p in [0.1]:
#     results_overall[p]=[]
#     results_polys[p]=[]
#     model = PPO.load(MODLE_FILE+"-200000-"+str(p)+".zip")
#     model.set_env(env)
#     print("p=",p)
    
#     print("Overall:")
#     env.unwrapped.init_polytopes(1.0,[])
#     mean_reward, std_reward, nb_crashes = evaluate_policy2(model.policy, env, n_eval_episodes=eval_episode_length)
#     results_overall[p].append((mean_reward, std_reward, nb_crashes))
#     print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}; crashes: {nb_crashes}")
    
#     print("Focus Polytopes:")
#     env.unwrapped.init_polytopes(0.0,retrain_polytopes)
#     mean_reward, std_reward, nb_crashes = evaluate_policy2(model.policy, env, n_eval_episodes=eval_episode_length)
#     results_polys[p].append((mean_reward, std_reward, nb_crashes))
#     print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}; crashes: {nb_crashes}")


# print(results_overall)
# print(results_polys)


# p= 0.1 and network bigger:
# Overall:
# mean_reward:3228.40 +/- 1757.96
# Focus Polytopes:
# mean_reward:3098.52 +/- 2133.65
#Another run
# p= 0.1
# Overall:
# mean_reward:3266.17 +/- 918.52; crashes: 12
# Focus Polytopes:
# mean_reward:709.82 +/- 2726.75; crashes: 502


# p= 0.1 and network smaller:
# Overall:
# mean_reward:2335.72 +/- 2130.89; crashes: 0
# Focus Polytopes:
# mean_reward:3481.49 +/- 1469.80; crashes: 67


onnxable_model = OnnxableActionPolicy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net)
# onnxable_model.graph.output[0].name = "out1"
# onnxable_model.graph.node[len(onnxable_model.graph.node)-1].output[0]="out1"
dummy_input = torch.randn(1, 2)
torch.onnx.export(onnxable_model, dummy_input, f"{MODLE_FILE}-{RUN_NB}.onnx", opset_version=9, dynamo=False)