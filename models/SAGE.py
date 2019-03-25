import torch
import torch
import torch.nn.functional as F
from boxx import timeit
from torch import nn
from torch_geometric.nn import global_add_pool
from nn import DIFFPool

from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, data, writer, dropout=0):
        super(SAGE, self).__init__()
        self.writer = writer
        self.num_features = data.num_features
        self.edge_attr_dim = data.edge_attr.shape[-1]
        self.B = data.y.shape[0]
        self.num_nodes = int(data.num_nodes / self.B)

        self.conv1 = SAGEConv(self.num_features, 30)

        self.pconv1 = SAGEConv(30, 32)
        self.pool1 = DIFFPool()

        self.pconv2 = SAGEConv(30, 4)
        self.pool2 = DIFFPool()

        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(30 * 4, 32)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, edge_index, edge_attr, adj):
        if x.dim() == 3:
            x, edge_index, edge_attr, y = \
                x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0), y.squeeze(0)

        if adj.dim() == 3:
            adj = adj.squeeze(0)

        x = self.conv1(x, edge_index)
        self.writer.add_histogram('conv1_x_std', x.std(dim=0))

        s = self.pconv1(x, edge_index)
        x, edge_index, edge_attr, adj, reg1 = self.pool1(x, adj, s)
        self.writer.add_histogram('pool1_x_std', x.std(dim=0))

        s = self.pconv2(x, edge_index)
        x, edge_index, edge_attr, adj, reg2 = self.pool2(x, adj, s)
        self.writer.add_histogram('pool2_x_std', x.std(dim=0))

        x = x.view(self.B, -1)
        x = self.drop2(F.relu(self.fc1(self.drop1(x))))
        x = self.fc2(x)

        # reg = torch.tensor([0], dtype=torch.float, device=x.device)
        reg = reg1 * 10 + reg2 * 0.1
        return x, reg