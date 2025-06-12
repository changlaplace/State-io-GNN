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

gamma = 0.99
clip_eps = 0.05
ppo_epochs = 4
train_node_num = 30
train_env_num = 1
episode_num = 30

logger = setup_logger(log_prefix=f"get_model_{train_node_num}")

policy = GNNPolicy(in_channels=4, edge_feat_dim=4, hidden_dim=64, use_attention=False)

optimizer = Adam(policy.parameters(), lr=3e-4)

for i in range(train_env_num):
    current_seed = random.randint(0, 1000000)
    logger.warning(f"Now training with the {i}th env node {train_node_num} with seed {current_seed}")
    train_env = StateIOEnv(renderflag=False, num_nodes=train_node_num, seed=current_seed)
    train_ppo(train_env, policy, optimizer, episode_num, logger = logger)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join("models", f"train_node_num{train_node_num}_{timestamp}.pt")
torch.save(policy.state_dict(), model_path)
logger.info(f"Trained model saved to {model_path}")