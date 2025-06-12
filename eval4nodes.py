# main.py
import os
import torch
from torch.optim import Adam
from stateio_gym.stateio_env import StateIOEnv
from gnn_policy import GNNPolicy
from ppo_agent import select_action, compute_returns
import datetime
import logging
from ppo_train import *


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_folder = "./models" 
model_name = "train_node_num20_20250611_212420.pt" 

gamma = 0.99
clip_eps = 0.05
ppo_epochs = 4
test_node_nums = [2, 3, 4, 5, 6, 7, 8, 9]
test_env_num = 3
episode_num = 3

def evaluate_policy_w_nodes(policy, test_node_nums, test_env_num, episode_num, random_bench=False):
    policy.eval()
    reward_w_node = []
    time_w_node = []
    for num_node in test_node_nums:
        rewards = []
        times = []
        for env_num in range(test_env_num):
            current_seed = random.randint(0, 1000000)
            env = StateIOEnv(renderflag=False, num_nodes=num_node, seed = current_seed) 
            for epi in range(episode_num):
                obs, _ = env.reset()
                done = False
                truncated = False
                episode_reward = 0
                step_count = 0
                while not (done or truncated):
                    with torch.no_grad():
                        if not random_bench:
                            logits = policy(obs)
                            probs = torch.softmax(logits, dim=0)
                            dist = torch.distributions.Categorical(probs)
                            action_index = dist.sample()
                            src = obs.edge_index[0, action_index].item()
                            dst = obs.edge_index[1, action_index].item()
                        elif random_bench:
                            num_edges = obs.edge_index.shape[1]
                            action_index = np.random.randint(0, num_edges)
                            src = obs.edge_index[0, action_index].item()
                            dst = obs.edge_index[1, action_index].item()
                        obs, reward, done, truncated, _ = env.step((src, dst))
                        episode_reward += reward
                        step_count += 1

                status = "done" if done else "truncated"
                
                msg = f"[Eval Episode {epi}] Total reward: {episode_reward:.2f}, Steps: {step_count}, Terminated by: {status}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg) 
                rewards.append(episode_reward)
                times.append(step_count)               

        reward_w_node.append(rewards)
        time_w_node.append(times)
    return reward_w_node, time_w_node


logger = setup_logger(log_prefix=f"eval_model_{timestamp}")

policy = GNNPolicy(in_channels=4, edge_feat_dim=4, hidden_dim=64, use_attention=False)
policy.load_state_dict(torch.load(os.path.join(model_folder, model_name)))
policy.eval()
import matplotlib.pyplot as plt
import numpy as np


reward_w_node, time_w_node = evaluate_policy_w_nodes(policy,test_node_nums,test_env_num,episode_num)
reward_w_node_r, time_w_node_r = evaluate_policy_w_nodes(policy,test_node_nums,test_env_num,episode_num,random_bench=True)

reward_mean = [np.mean(x) for x in reward_w_node]
reward_std = [np.std(x) for x in reward_w_node]
time_mean = [np.mean(x) for x in time_w_node]
time_std = [np.std(x) for x in time_w_node]

reward_mean_r = [np.mean(x) for x in reward_w_node_r]
reward_std_r = [np.std(x) for x in reward_w_node_r]
time_mean_r = [np.mean(x) for x in time_w_node_r]
time_std_r = [np.std(x) for x in time_w_node_r]

image_folder = "./images"
plt.figure()
plt.errorbar(test_node_nums, reward_mean, yerr=reward_std, fmt='-o', capsize=4, label='Policy', color='orange')
plt.errorbar(test_node_nums, reward_mean_r, yerr=reward_std_r, fmt='-o', capsize=4, label='Random', color='gray')
plt.xlabel("Number of Nodes")
plt.ylabel("Total Reward")
plt.title("Reward vs Node Count")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(image_folder, f"Reward vs node {timestamp}"))
plt.show()
plt.close()


plt.figure()
plt.errorbar(test_node_nums, time_mean, yerr=time_std, fmt='-o', capsize=4, label='Policy', color='orange')
plt.errorbar(test_node_nums, time_mean_r, yerr=time_std_r, fmt='-o', capsize=4, label='Random', color='gray')
plt.xlabel("Number of Nodes")
plt.ylabel("Steps to Completion")
plt.title("Steps vs Node Count")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(image_folder, f"Steps vs node {timestamp}"))
plt.show()
plt.close()