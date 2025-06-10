# main.py
import os
import torch
from torch.optim import Adam
from stateio_gym.stateio_env import StateIOEnv
from gnn_policy import GNNPolicy
from ppo_agent import select_action, compute_returns
import datetime


IFSAVE = True
IFLOAD = False


policy = GNNPolicy(in_channels=4, edge_feat_dim=4, hidden_dim=64, use_attention=True)
if IFLOAD:
    policy.load_state_dict(torch.load("policy_gnn.pt"))

env = StateIOEnv(renderflag=False, num_nodes=30, seed=42)
optimizer = Adam(policy.parameters(), lr=3e-4)

gamma = 0.99
clip_eps = 0.2
ppo_epochs = 4


for episode in range(10):
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

if IFSAVE:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"policy_gnn_{timestamp}.pt"
    model_path = os.path.join(r'./models', model_path)
    torch.save(policy.state_dict(), model_path)
    print(f"GNN policy model saved to {model_path}")
