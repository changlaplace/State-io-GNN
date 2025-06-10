# main.py
import os
import torch
from torch.optim import Adam
from stateio_gym.stateio_env import StateIOEnv
from gnn_policy import GNNPolicy
from ppo_agent import select_action, compute_returns
import datetime
import logging

def setup_logger(log_dir="./logs", log_prefix="log"):

    os.makedirs(log_dir, exist_ok=True)
    pid = os.getpid()
    log_filename = os.path.join(log_dir, f"{log_prefix}_pid{pid}.log")

    logger = logging.getLogger(str(pid))
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        formatter = logging.Formatter('[%(asctime)s] [PID %(process)d] [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

gamma = 0.99
clip_eps = 0.2
ppo_epochs = 4

def train_ppo(env, policy, optimizer, episode_num):
    for episode in range(episode_num):
        obs, _ = env.reset()
        log_probs, rewards, entropies, edge_ids = [], [], [], []
        done = False
        truncated = False
        data_list = []

        while not (done or truncated):
            data = obs  # now obs is torch_geometric.data.Data
            action, log_prob, entropy, edge_id = select_action(policy, data)

            obs, reward, done, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            edge_ids.append(edge_id)
            rewards.append(reward)
            entropies.append(entropy)
            data_list.append(data)

        returns = compute_returns(rewards, [done or truncated] * len(rewards))

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        advantages = returns - returns.mean()
        
        log_probs = log_probs.detach()
        advantages = advantages.detach()
        entropies = entropies.detach()
        for _ in range(ppo_epochs):
            for i, data in enumerate(data_list):
                logits = policy(data)
                probs = torch.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                new_log_prob = dist.log_prob(edge_ids[i])  # use the same sampled edge id
                ratio = torch.exp(new_log_prob - log_probs[i].detach())

                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[i]
                loss = -torch.min(surr1, surr2) - 0.01 * entropies[i]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        status = "done" if done else "truncated"
        print(f"[Episode {episode}] Total reward: {sum(rewards):.2f}, Time step: {env.step_count}, Terminated by: {status}")

def evaluate_policy(env, policy, episode_num=100):
    policy.eval()
    total_rewards = []

    for episode in range(episode_num):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        while not (done or truncated):
            with torch.no_grad():
                logits = policy(obs)
                probs = torch.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action_index = dist.sample()
                src = obs.edge_index[0, action_index].item()
                dst = obs.edge_index[1, action_index].item()
                obs, reward, done, truncated, _ = env.step((src, dst))
                episode_reward += reward
                step_count += 1

        total_rewards.append(episode_reward)
        status = "done" if done else "truncated"
        print(f"[Eval Episode {episode}] Total reward: {episode_reward:.2f}, Steps: {step_count}, Terminated by: {status}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n✅ Evaluation completed over {episode_num} episodes, average reward: {avg_reward:.2f}")
    return avg_reward


if __name__=="__main__":

    IFSAVE = True
    IFLOAD = False


    policy = GNNPolicy(in_channels=4, edge_feat_dim=4, hidden_dim=64, use_attention=True)
    if IFLOAD:
        policy.load_state_dict(torch.load("policy_gnn.pt"))

    env = StateIOEnv(renderflag=False, num_nodes=30, seed=42)
    optimizer = Adam(policy.parameters(), lr=3e-4)
    
    train_ppo(env, policy, optimizer, episode_num=100)

    if IFSAVE:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"policy_gnn_{timestamp}.pt"
        model_path = os.path.join(r'./models', model_path)
        torch.save(policy.state_dict(), model_path)
        print(f"GNN policy model saved to {model_path}")
