import os
import torch
from torch.optim import Adam
from stateio_gym.stateio_env import StateIOEnv
from gnn_policy import GNNPolicy
from ppo_agent import select_action, compute_returns
import datetime


IFSAVE = True
IFLOAD = False

model_folder = r'./models'
model_name = r'policy_gnn_20250610_000721.pt'


policy = GNNPolicy(in_channels=4, edge_feat_dim=4, hidden_dim=64)
if IFLOAD:
    policy.load_state_dict(torch.load(os.path.join(model_folder, model_name)))
