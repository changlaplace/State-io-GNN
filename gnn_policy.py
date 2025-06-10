# gnn_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNPolicy(nn.Module):
    # in_channel is the node features dim [mytroop, neutraltroop, posx, posy]
    # edge_feat_dim is 4 [distance, total_units, avg_time, num_transfers]
    def __init__(self, in_channels = 4, edge_feat_dim = 4, hidden_dim = 64):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        # Currently x has the dimension of [num_nodes, hidden_dim]
        src, dst = edge_index
        
        h_src = x[src]
        h_dst = x[dst]
        h_diff = torch.abs(h_src - h_dst)  
        edge_input = torch.cat([h_diff, edge_attr], dim=1)

        logits = self.edge_mlp(edge_input).squeeze(-1)
        return logits  # [num_edges]
