import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_sort_pool

from nn import SortPool, DIFFPool


class GCNConvSortPool(torch.nn.Module):
    """
    designed for multi-channel graph in DGCNN_SortPool
    """

    def __init__(self, in_channels, k):
        super(GCNConvSortPool, self).__init__()

        self.gcnconv1 = GCNConv(in_channels, 1)
        self.gcnconv2 = GCNConv(1, 1)
        # self.gcnconv3 = GCNConv(1, 1)
        # self.gcnconv4 = GCNConv(1, 1)
        # self.gcnconv5 = GCNConv(1, 1)
        self.sortpool = SortPool(k)
        self.conv1d1 = nn.Conv1d(2, 3, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv1d2 = nn.Conv1d(3, 1, kernel_size=3, stride=1)
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
        # x = torch.cat([x, self.gcnconv3(x[:, -1].view(-1, 1), edge_index, edge_attr)], dim=-1)
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


# real-DGCNN
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

        return x


# Note: grad_fn doesn't work
class BUGGY_DGCNN1(torch.nn.Module):
    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(DGCNN, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.channels = data.edge_attr.shape[-1]
        self.gcn_conv_level = 5
        # multi-dimensional edge_attr is implemented as separate channels and concatenated before dense layer.
        for channel in range(self.channels):
            for conv_level in range(1, self.gcn_conv_level + 1):
                exec("self.gcnconv{}_channel{} = GCNConv({}, 1)".format(
                    conv_level, channel, 1 if conv_level > 1 else self.num_features
                ))
            exec("self.conv1d1_channel{} = nn.Conv1d(5, 3, kernel_size=3, stride=1)".format(channel))
            exec("self.maxpool1_channel{} = nn.MaxPool1d(kernel_size=3, stride=1)".format(channel))
            exec("self.conv1d2_channel{} = nn.Conv1d(3, 1, kernel_size=3, stride=1)".format(channel))
            exec("self.maxpool2_channel{} = nn.MaxPool1d(kernel_size=3, stride=1)".format(channel))

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
        if x.dim() == 3:  # a batch
            if x.shape[0] != 1:
                raise Exception("batch size greater than 1 is not supported.")
            x, edge_index, edge_attr = x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0)

        for i, channel in enumerate(range(self.channels)):
            e = edge_attr[:, i]
            # gcn conv
            for conv_level in range(1, self.gcn_conv_level + 1):
                conv = getattr(self, "gcnconv{}_channel{}".format(conv_level, channel))
                if conv_level == 1:
                    xx = conv(x, edge_index, e)
                else:
                    xx = torch.cat([xx, conv(xx[:, -1].view(-1, 1), edge_index, e)], dim=-1)
            # sort pool
            N, D = xx.size()
            xx = global_sort_pool(x=xx,
                                  batch=torch.tensor(
                                      [0 for i in range(self.num_nodes)], dtype=torch.long, device=xx.device
                                  ), k=self.num_nodes)
            xx = xx.view(-1, N, D).permute(0, 2, 1)
            # conv and pool
            xx = getattr(self, "conv1d1_channel{}".format(channel))(xx)
            xx = getattr(self, "maxpool1_channel{}".format(channel))(xx)
            xx = getattr(self, "conv1d2_channel{}".format(channel))(xx)
            xx = getattr(self, "maxpool2_channel{}".format(channel))(xx)

            all_x = xx if i == 0 else torch.cat([all_x, xx], dim=0)

        x = all_x.view(1, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        print(x)
        return x


# Note: Parameters in GCNConv is not placed to CUDA
class BUGGY_DGCNN2(torch.nn.Module):
    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(DGCNN, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.channels = data.edge_attr.shape[-1]
        self.gcn_conv_level = 5
        # multi-dimensional edge_attr is implemented as separate channels and concatenated before dense layer.
        self.gcnconv1 = [GCNConv(self.num_features, 1) for _ in range(self.channels)]
        self.gcnconv2 = [GCNConv(1, 1) for _ in range(self.channels)]
        self.gcnconv3 = [GCNConv(1, 1) for _ in range(self.channels)]
        self.gcnconv4 = [GCNConv(1, 1) for _ in range(self.channels)]
        self.gcnconv5 = [GCNConv(1, 1) for _ in range(self.channels)]
        self.conv1d1 = [nn.Conv1d(5, 3, kernel_size=3, stride=1) for _ in range(self.channels)]
        self.maxpool1 = [nn.MaxPool1d(kernel_size=3, stride=1) for _ in range(self.channels)]
        self.conv1d2 = [nn.Conv1d(3, 1, kernel_size=3, stride=1) for _ in range(self.channels)]
        self.maxpool2 = [nn.MaxPool1d(kernel_size=3, stride=1) for _ in range(self.channels)]
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
        print(getattr(self, "conv1d1_channel1"))
        if x.dim() == 3:  # a batch
            if x.shape[0] != 1:
                raise Exception("batch size greater than 1 is not supported.")
            x, edge_index, edge_attr = x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0)

        for i, channel in enumerate(range(self.channels)):
            e = edge_attr[:, i]
            xx = self.gcnconv1[channel](x, edge_index, e)
            xx = torch.cat([xx, self.gcnconv2[channel](xx[:, -1].unsqueeze(-1), edge_index, e)], dim=-1)
            xx = torch.cat([xx, self.gcnconv3[channel](xx[:, -1].unsqueeze(-1), edge_index, e)], dim=-1)
            xx = torch.cat([xx, self.gcnconv4[channel](xx[:, -1].unsqueeze(-1), edge_index, e)], dim=-1)
            xx = torch.cat([xx, self.gcnconv5[channel](xx[:, -1].unsqueeze(-1), edge_index, e)], dim=-1)
            N, D = xx.size()
            xx = global_sort_pool(x=xx, batch=torch.tensor([0 for i in range(self.num_nodes)], dtype=torch.long),
                                  k=self.num_nodes)
            xx = xx.view(-1, N, D).permute(0, 2, 1)
            xx = self.conv1d1[channel](xx)
            xx = self.maxpool1[channel](xx)
            xx = self.conv1d2[channel](xx)
            xx = self.maxpool2[channel](xx)

            all_x = xx if i == 0 else torch.cat([all_x, xx], dim=0)

        x = all_x.view(1, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        return x
