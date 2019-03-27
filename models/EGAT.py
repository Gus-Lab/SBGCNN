import torch
import torch.nn.functional as F
from boxx import timeit
from torch import nn
from torch_geometric.nn import global_add_pool
from nn import DIFFPool

from nn import MEGATConv, EGATConv


class _EGATConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 num_nodes,
                 batch_size,
                 writer
                 ):
        super(_EGATConv, self).__init__()
        self.writer = writer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.B = batch_size

        self.conv1 = EGATConv(in_channels, 6, heads=5, dropout=dropout, concat=True)

        self.pconv1 = EGATConv(30, 32)
        self.pool1 = DIFFPool()

        self.pconv2 = EGATConv(30, 4)
        self.pool2 = DIFFPool()


    def forward(self, x, edge_index, edge_attr, adj):
        x, edge_index, e = self.conv1(x, edge_index, edge_attr, save=True)
        e = self.dot(e).unsqueeze(-1)
        self.writer.add_histogram('conv1_x_std', x.std(dim=0))

        s, _, _ = self.pconv1(x, edge_index, edge_attr)
        x, edge_index, edge_attr, adj, reg1 = self.pool1(x, adj, s)
        self.writer.add_histogram('pool1_x_std', x.std(dim=0))

        s, _, _ = self.pconv2(x, edge_index, edge_attr)
        x, edge_index, edge_attr, adj, reg2 = self.pool2(x, adj, s)
        self.writer.add_histogram('pool2_x_std', x.std(dim=0))

        # reg = torch.tensor([0], dtype=torch.float, device=x.device)
        reg = reg1 * 10 + reg2 * 0.1
        return x, reg

    @staticmethod
    def dot(e):
        es = torch.unbind(e, dim=-1)
        for i, t in enumerate(es):
            new_e = t if i == 0 else new_e
            new_e = new_e * t if i != 0 else new_e
        return new_e


class EGAT(torch.nn.Module):
    def __init__(self, data, writer, dropout=0):
        super(EGAT, self).__init__()
        self.writer = writer
        self.num_features = data.num_features
        self.edge_attr_dim = data.edge_attr.shape[-1]
        self.B = data.y.shape[0]
        self.num_nodes = int(data.num_nodes / self.B)

        self.egatconv_channel1 = _EGATConv(self.num_features, 4, dropout, self.num_nodes, self.B, self.writer)

        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(30 * 4, 32)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, edge_index, edge_attr, y, adj):
        if x.dim() == 3:
            x, edge_index, edge_attr, y = \
                x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0), y.squeeze(0)

        if adj.dim() == 3:
            adj = adj.squeeze(0)

        x, reg = self.egatconv_channel1(x, edge_index, edge_attr, adj)
        x = x.view(self.B, -1)
        x = self.drop2(F.relu(self.fc1(self.drop1(x))))
        x = self.fc2(x)

        # reg = torch.tensor([0], dtype=torch.float, device=x.device)
        return x, reg
