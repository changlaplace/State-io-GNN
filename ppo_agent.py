# ppo_agent.py

import torch
import torch.nn.functional as F
import torch.distributions as D

def select_action(policy, data):
    logits = policy(data)
    probs = F.softmax(logits, dim=0)
    dist = D.Categorical(probs)
    edge_id = dist.sample()
    action = (data.edge_index[0, edge_id].item(), data.edge_index[1, edge_id].item())
    return action, dist.log_prob(edge_id), dist.entropy(), edge_id

def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    R = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - done)
        returns.insert(0, R)
    return torch.tensor(returns)
