import torch
import torch.nn.functional as F
from torch import nn
from nn import DIFFPool

from boxx import timeit

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, remove_self_loops

from torch_geometric.nn.inits import glorot, zeros

from utils import add_self_loops_with_edge_attr


class SGCN(torch.nn.Module):
    def __init__(self, data, writer, dropout=0):
        super(SGCN, self).__init__()
        self.writer = writer
        self.num_features = data.num_features
        self.edge_attr_dim = data.edge_attr.shape[-1]
        self.B = data.y.shape[0]
        self.num_nodes = int(data.num_nodes / self.B)

        self.gcnconv1 = GCNConv(data.num_features, 4)
        # self.bn1 = nn.BatchNorm1d(4)

        self.fc1 = nn.Linear(self.num_nodes * 4, 32)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, edge_index, edge_attr, y, adj):
        if x.dim() == 3:
            x, edge_index, edge_attr, y = \
                x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0), y.squeeze(0)

        if adj.dim() == 3:
            adj = adj.squeeze(0)

        x = self.gcnconv1(x, edge_index, edge_attr)
        self.writer.add_histogram('conv1_x_std', x.std(dim=0))
        # x = self.bn1(x)
        x = x.view(self.B, -1)
        x = self.drop2(F.relu(self.fc1(self.drop1(x))))
        x = self.fc2(x)

        reg = torch.tensor([0], dtype=torch.float, device=x.device)
        return x, reg

# class GCNConv(torch.nn.Module):
#     """
#     designed for multi-channel graph in DGCNN_SortPool
#     """
#
#     def __init__(self, in_channels, num_nodes):
#         super(GCNConv, self).__init__()
#
#         self.gcnconv1 = UWGCNConv(in_channels, 7)
#         self.gcnconv2 = UWGCNConv(7, 7)
#         # self.gcnconv3 = GCNConv(7, 7)
#         self.diffpool1 = DIFFPool(num_nodes, 64)
#         # self.diffpool2 = DIFFPool(64, 32)
#         # self.diffpool3 = DIFFPool(32, 1)
#         # self.diffpool4 = DIFFPool(4, 1)
#
#     def forward(self, x, edge_index, edge_attr, adj):
#         """
#         :param x: Node feature matrix with shape [num_nodes, num_node_features]
#         :param edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
#         :param edge_attr: Edge feature matrix with shape [num_edges, 1]
#         :param adj: Adjacency matrix with shape [num_nodes, num_nodes]
#         :return:
#         """
#         x = self.gcnconv1(x, edge_index, edge_attr)
#         x = self.gcnconv2(x, edge_index, edge_attr)
#         return x


class GCNDP(torch.nn.Module):

    def __init__(self,
                 data,
                 writer,
                 dropout=0
                 ):
        super(GCNDP, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.channels = data.edge_attr.shape[-1]
        self.B = data.y.shape[0]

        # multi-dimensional edge_attr is implemented as separate channels and concatenated before dense layer.
        self.gcnconvdiffpool_channel1 = GCNConv(self.num_features, num_nodes=self.num_nodes)
        self.gcnconvdiffpool_channel2 = GCNConv(self.num_features, num_nodes=self.num_nodes)
        self.gcnconvdiffpool_channel3 = GCNConv(self.num_features, num_nodes=self.num_nodes)
        # self.pool1 = nn.AvgPool1d(kernel_size=self.channels)
        # self.conv1d = nn.Conv1d(129, 4, 3)
        self.fc1 = nn.Linear(129 * 7 * 3, 8)
        self.drop1 = nn.Dropout(self.dropout)
        # self.fc2 = nn.Linear(8, 8)
        # self.drop2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x, edge_index, edge_attr, adj, *args, **kwargs):
        """
        multi-dimensional edge_attr is implemented as separate channels and concatenated before dense layer.
        :param x: Node feature matrix with shape [num_nodes, num_node_features]
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        :param edge_attr: Edge feature matrix with shape [num_edges, num_edge_features], num_edge_features >= 1
        :param adj: Adjacency matrix with shape [num_edge_features, num_nodes, num_nodes]
        :param args:
        :param kwargs:
        :return:
        """
        if x.dim() == 3:  # one batch
            if x.shape[0] != 1:
                raise Exception("batch size greater than 1 is not supported.")
            x, edge_index, edge_attr, adj = x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0), adj.squeeze(0)
        x1, reg1 = self.gcnconvdiffpool_channel1(x, edge_index, edge_attr[:, 0], adj[0, :, :])
        x2, reg2 = self.gcnconvdiffpool_channel2(x, edge_index, edge_attr[:, 1], adj[1, :, :])
        x3, reg3 = self.gcnconvdiffpool_channel3(x, edge_index, edge_attr[:, 2], adj[2, :, :])
        x = torch.cat([x1, x2, x3], dim=-1).unsqueeze(0)
        # x = self.conv1d(x)
        x = x.view(1, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        # x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        reg = reg1 + reg2 + reg3
        return x, reg


class WGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(WGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        loop_weight = torch.full(
            (x.size(0),),
            1 if not self.improved else 2,
            dtype=x.dtype,
            device=x.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]

        x = torch.matmul(x, self.weight)
        return self.propagate('add', edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class UWGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(UWGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        # if edge_weight is None:
        #     edge_weight = torch.ones(
        #         (edge_index.size(1),), dtype=x.dtype, device=x.device)
        # edge_weight = edge_weight.view(-1)
        # assert edge_weight.size(0) == edge_index.size(1)

        # Add self-loops to adjacency matrix.
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops_with_edge_attr(edge_index, edge_weight, num_nodes=x.size(0))

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('nan')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]

        # x = torch.matmul(x, self.weight)
        # print(norm)
        return self.propagate('add', edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
