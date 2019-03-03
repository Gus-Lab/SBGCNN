import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from nn import SortPool


class GCNConvSortPool(torch.nn.Module):
    """
    designed for multi-channel graph in DGCNN_SortPool
    """

    def __init__(self, in_channels, k):
        super(GCNConvSortPool, self).__init__()

        self.gcnconv1 = GCNConv(in_channels, 1)
        self.gcnconv2 = GCNConv(1, 1)
        self.gcnconv3 = GCNConv(1, 1)
        # self.gcnconv4 = GCNConv(1, 1)
        # self.gcnconv5 = GCNConv(1, 1)
        self.sortpool = SortPool(k)
        self.conv1d1 = nn.Conv1d(3, 1, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv1d2 = nn.Conv1d(1, 1, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1)

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: Node feature matrix with shape [num_nodes, num_node_features]
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        :param edge_attr: Edge feature matrix with shape [num_edges, 1]
        :return:
        """
        # GCNConv
        x = self.gcnconv1(x, edge_index, edge_attr)
        x = torch.cat([x, self.gcnconv2(x[:, -1].view(-1, 1), edge_index, edge_attr)], dim=-1)
        x = torch.cat([x, self.gcnconv3(x[:, -1].view(-1, 1), edge_index, edge_attr)], dim=-1)
        # x = torch.cat([x, self.gcnconv4(x[:, -1].view(-1, 1), edge_index, edge_attr)], dim=-1)
        # x = torch.cat([x, self.gcnconv5(x[:, -1].view(-1, 1), edge_index, edge_attr)], dim=-1)
        # Sort Pooling
        N, D = x.size()
        x = self.sortpool(x)
        x = x.view(-1, N, D).permute(0, 2, 1)
        # Conv1d & Pool
        x = self.conv1d1(x)
        x = self.maxpool1(x)
        x = self.conv1d2(x)
        x = self.maxpool2(x)

        return x


class DGCNN(torch.nn.Module):

    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(DGCNN, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.channels = data.edge_attr.shape[-1]

        # multi-dimensional edge_attr is implemented as separate channels and concatenated before dense layer.
        self.dgcnnconv_channel1 = GCNConvSortPool(self.num_features, k=self.num_nodes)
        self.dgcnnconv_channel2 = GCNConvSortPool(self.num_features, k=self.num_nodes)
        self.dgcnnconv_channel3 = GCNConvSortPool(self.num_features, k=self.num_nodes)

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

        x1 = self.dgcnnconv_channel1(x, edge_index, edge_attr[:, 0])
        x2 = self.dgcnnconv_channel1(x, edge_index, edge_attr[:, 1])
        x3 = self.dgcnnconv_channel1(x, edge_index, edge_attr[:, 2])
        all_x = torch.cat([x1, x2, x3], dim=-2)

        x = all_x.view(1, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        reg = torch.tensor([0.0], device=x.device)
        return x, reg
