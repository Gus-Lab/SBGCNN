import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_sort_pool

from nn import SortPool, DIFFPool


class GCNConvDiffPool(torch.nn.Module):
    """
    designed for multi-channel graph in DGCNN_SortPool
    """

    def __init__(self, in_channels, num_nodes):
        super(GCNConvDiffPool, self).__init__()

        self.gcnconv1 = GCNConv(in_channels, 1)
        self.gcnconv2 = GCNConv(1, 1)
        self.gcnconv3 = GCNConv(1, 1)
        self.diffpool1 = DIFFPool(num_nodes, 48)
        self.diffpool2 = DIFFPool(48, 12)
        self.diffpool3 = DIFFPool(12, 4)
        self.diffpool4 = DIFFPool(4, 1)

    def forward(self, x, edge_index, edge_attr, adj):
        """
        :param x: Node feature matrix with shape [num_nodes, num_node_features]
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        :param edge_attr: Edge feature matrix with shape [num_edges, 1]
        :param adj: Adjacency matrix with shape [num_nodes, num_nodes]
        :return:
        """
        x = self.gcnconv1(x, edge_index, edge_attr)
        x = torch.cat([x, self.gcnconv2(x[:, -1].view(-1, 1), edge_index, edge_attr)], dim=-1)
        x = torch.cat([x, self.gcnconv2(x[:, -1].view(-1, 1), edge_index, edge_attr)], dim=-1)
        # Differentiable Pooling
        N, D = x.size()
        x, edge_index, edge_attr, adj, reg1 = self.diffpool1(x)
        x, edge_index, edge_attr, adj, reg2 = self.diffpool1(x)
        x, edge_index, edge_attr, adj, reg3 = self.diffpool1(x)
        x, edge_index, edge_attr, adj, reg4 = self.diffpool1(x)

        reg = reg1 + reg2 + reg3 + reg4
        return x, reg


class GCNDP(torch.nn.Module):

    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(GCNDP, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.channels = data.edge_attr.shape[-1]

        # multi-dimensional edge_attr is implemented as separate channels and concatenated before dense layer.
        self.gcnconvdiffpool1_channel1 = GCNConvDiffPool(self.num_features, num_nodes=self.num_nodes)
        self.gcnconvdiffpool2_channel1 = GCNConvDiffPool(self.num_features, num_nodes=self.num_nodes)
        self.gcnconvdiffpool3_channel1 = GCNConvDiffPool(self.num_features, num_nodes=self.num_nodes)

        self.fc1 = nn.Linear(self.channels * 121, 32)
        self.drop1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(32, 6)
        self.drop2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(6, 2)

    def forward(self, x, edge_index, edge_attr, *args, **kwargs):
        """
        multi-dimensional edge_attr is implemented as separate channels and concatenated before dense layer.
        :param x: Node feature matrix with shape [num_nodes, num_node_features]
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        :param edge_attr: Edge feature matrix with shape [num_edges, num_edge_features], num_edge_features >= 1
        :param args:
        :param kwargs:
        :return:
        """
        if x.dim() == 3:  # one batch
            if x.shape[0] != 1:
                raise Exception("batch size greater than 1 is not supported.")
            x, edge_index, edge_attr = x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0)

        x1, reg1 = self.dgcnnconv_channel1(x, edge_index, edge_attr[:, 0])
        x2, reg2 = self.dgcnnconv_channel1(x, edge_index, edge_attr[:, 1])
        x3, reg3 = self.dgcnnconv_channel1(x, edge_index, edge_attr[:, 2])
        all_x = torch.cat([x1, x2, x3], dim=-2)

        x = all_x.view(1, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        reg = reg1 + reg2 + reg3
        return x, reg