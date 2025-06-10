import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GNNPolicy(nn.Module):
    def __init__(self, in_channels=4, edge_feat_dim=4, hidden_dim=64, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.conv1 = GATConv(in_channels, hidden_dim, heads=1, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        else:
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2+ edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]

        h_diff = torch.cat([h_src, h_dst],dim=1)
        edge_input = torch.cat([h_diff, edge_attr], dim=1)

        logits = self.edge_mlp(edge_input).squeeze(-1)
        return logits  # [num_edges]
