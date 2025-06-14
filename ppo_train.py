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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    pid = os.getpid()
    log_filename = os.path.join(log_dir, f"{timestamp}_{log_prefix}.log")

    logger = logging.getLogger(str(pid))
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        formatter = logging.Formatter('[%(asctime)s] [PID %(process)d] [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler (terminal)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

gamma = 0.99
clip_eps = 0.05
ppo_epochs = 4

def train_ppo(env, policy, optimizer, episode_num, logger=None, update_every=10):
    reward_list = []
    times_list = []

    gamma = 0.99
    eps_clip = 0.05
    entropy_coeff = 0.01

    # 用于收集数据的 buffer
    buffer = []

    for episode in range(episode_num):
        obs, _ = env.reset()
        done, truncated = False, False
        episode_reward = 0
        step_count = 0

        while not (done or truncated):
            data = obs
            action, log_prob, entropy, edge_id = select_action(policy, data)

            # detach edge_id（如果是 Tensor）
            edge_id = edge_id.item() if isinstance(edge_id, torch.Tensor) else edge_id

            obs, reward, done, truncated, _ = env.step(action)

            buffer.append({
                "data": data,
                "action": edge_id,
                "log_prob": log_prob.detach(),
                "reward": reward,
                "entropy": entropy.detach(),
            })

            episode_reward += reward
            step_count += 1

        reward_list.append(episode_reward)
        times_list.append(step_count)

        if logger:
            status = "done" if done else "truncated"
            logger.info(f"[Episode {episode}] Reward: {episode_reward:.2f}, Steps: {step_count}, Terminated by: {status}")
        else:
            print(f"[Episode {episode}] Reward: {episode_reward:.2f}, Steps: {step_count}")

        # 如果达到 update 周期，开始训练
        if (episode + 1) % update_every == 0 or episode == episode_num - 1:
            # === 计算 returns 和 advantages ===
            rewards = [entry["reward"] for entry in buffer]
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = returns - returns.mean()
            advantages /= (advantages.std() + 1e-8)

            # === PPO 多 epoch 更新 ===
            for _ in range(ppo_epochs):
                indices = torch.randperm(len(buffer))
                for i in indices:
                    entry = buffer[i]
                    data = entry["data"]
                    old_log_prob = entry["log_prob"]
                    edge_id = entry["action"]
                    advantage = advantages[i]
                    entropy = entry["entropy"]

                    logits = policy(data)
                    probs = torch.softmax(logits, dim=0)
                    dist = torch.distributions.Categorical(probs)
                    new_log_prob = dist.log_prob(torch.tensor(edge_id))

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) - entropy_coeff * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            buffer.clear()  # 清空 buffer，为下一轮收集数据

    return reward_list, times_list


# def train_ppo(env, policy, optimizer, episode_num, logger=None):
#     reward_list = []
#     times_list = []
#     for episode in range(episode_num):
#         obs, _ = env.reset()
#         log_probs, rewards, entropies, edge_ids = [], [], [], []
#         done = False
#         truncated = False
#         data_list = []

#         while not (done or truncated):
#             data = obs  # now obs is torch_geometric.data.Data
#             action, log_prob, entropy, edge_id = select_action(policy, data)

#             obs, reward, done, truncated, _ = env.step(action)

#             log_probs.append(log_prob)
#             edge_ids.append(edge_id)
#             rewards.append(reward)
#             entropies.append(entropy)
#             data_list.append(data)

#         returns = compute_returns(rewards, [done or truncated] * len(rewards))

#         log_probs = torch.stack(log_probs)
#         entropies = torch.stack(entropies)
#         advantages = returns - returns.mean()
        
#         log_probs = log_probs.detach()
#         advantages = advantages.detach()
#         entropies = entropies.detach()
#         for _ in range(ppo_epochs):
#             for i, data in enumerate(data_list):
#                 logits = policy(data)
#                 probs = torch.softmax(logits, dim=0)
#                 dist = torch.distributions.Categorical(probs)
#                 new_log_prob = dist.log_prob(edge_ids[i])  # use the same sampled edge id
#                 ratio = torch.exp(new_log_prob - log_probs[i].detach())

#                 surr1 = ratio * advantages[i]
#                 surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[i]
#                 loss = -torch.min(surr1, surr2) - 0.01 * entropies[i]

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#         status = "done" if done else "truncated"
#         msg = (f"[Episode {episode}] Total reward: {sum(rewards):.2f}, Time step: {env.step_count}, Terminated by: {status}")
#         if logger:
#             logger.info(msg)
#         else:
#             print(msg)
#         reward_list.append(rewards)
#         times_list.append(env.step_count)
#     return reward_list, times_list

def evaluate_policy(env, policy, episode_num=50, logger=None):
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
        
        msg = f"[Eval Episode {episode}] Total reward: {episode_reward:.2f}, Steps: {step_count}, Terminated by: {status}"
        if logger:
            logger.info(msg)
        else:
            print(msg)

    avg_reward = sum(total_rewards) / len(total_rewards)
    
    msg = f"\n✅ Evaluation completed over {episode_num} episodes, average reward: {avg_reward:.2f}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return avg_reward

import random

def transfer_experiment(train_node_num, test_node_num, use_attention=True, episode_num=10, eval_num=10):
    train_env_num = 5
    test_env_num = 5
    
    logger = setup_logger(log_dir='./logs',
                          log_prefix=(f"transfer{train_node_num}to{test_node_num}_"
                          f"trainenv{train_env_num}_testenv{test_env_num}_att{use_attention}"))

   
    

    policy = GNNPolicy(in_channels=4, edge_feat_dim=4, hidden_dim=64, use_attention=use_attention)
    optimizer = Adam(policy.parameters(), lr=3e-4)

    logger.info(f"🚀 Start training on {train_node_num} nodes, testing on {test_node_num} nodes")

    for i in range(train_env_num):
        current_seed = random.randint(0, 1000000)
        logger.warning(f"Now training with the {i}th env node {train_node_num} with seed {current_seed}")
        train_env = StateIOEnv(renderflag=False, num_nodes=train_node_num, seed=current_seed)
        train_ppo(train_env, policy, optimizer, episode_num, logger = logger)


    for j in range(test_env_num):
        current_seed = random.randint(0, 1000000)
        logger.warning(f"Now evaling with the {j}th env node {test_node_num} with seed {current_seed}")
        test_env = StateIOEnv(renderflag=False, num_nodes=test_node_num, seed = current_seed) 
        avg_reward = evaluate_policy(test_env, policy, episode_num=eval_num, logger = logger)
    
    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("models", f"transfer_{train_node_num}to{test_node_num}_{timestamp}.pt")
    torch.save(policy.state_dict(), model_path)
    logger.info(f"Trained model saved to {model_path}")
    logger.info(f"Avg reward on {test_node_num} nodes after training on {train_node_num} nodes: {avg_reward:.2f}")

    return avg_reward


if __name__=="__main__":    
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    node_a = 2
    node_b = 5
    
    print("========== Transfer: b → a ==========")
    transfer_experiment(train_node_num=node_b, test_node_num=node_a, use_attention=False)
   
    
    
