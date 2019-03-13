import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool


class Baseline(torch.nn.Module):

    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(Baseline, self).__init__()
        self.num_features = data.num_features
        self.B = data.y.shape[0]
        self.num_nodes = int(data.num_nodes / self.B)
        self.emb1 = nn.Linear(11, 2)
        # self.bn1 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(2 * self.num_nodes, 6)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(6, 2)
        # self.drop2 = nn.Dropout(dropout)
        # self.fc3 = nn.Linear(8, 2)

    def forward(self, x, edge_index, edge_attr, y, adj):
        if x.dim() == 3:
            x, edge_index, edge_attr, y = \
                x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0), y.squeeze(0)

        B = y.shape[0]
        x = self.emb1(x).view(B, -1)
        # x = x.view(B, -1)
        # x = global_add_pool(x, batch=torch.tensor([i for _ in range(self.num_nodes) for i in range(self.B)],
        #                                           device=x.device))
        x = F.relu(self.drop1(self.fc1(x)))
        # x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc2(x)

        reg = torch.tensor([0], dtype=torch.float, device=x.device)
        return x, reg
