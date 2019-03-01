import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import dense_diff_pool
from utils import adj_to_edge_index
from torch_geometric.nn.inits import uniform, glorot


class EGATConv(torch.nn.Module):
    """
    Adaptive Edge Features Graph Attentional Layer from the `"Adaptive Edge FeaturesGraph Attention Networks (GAT)"
    <https://arxiv.org/abs/1809.02709`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): Whether to concat or average multi-head
            attentions (default: :obj:`True`)
        negative_slope (float, optional): LeakyRELU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients, i.e. exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_attr_dim (int, required): The dimension of edge features. (default: :obj:`1`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 edge_attr_dim=1):
        super(EGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_attr_dim = edge_attr_dim

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, self.edge_attr_dim * out_channels))
        self.att_weight = nn.Parameter(torch.Tensor(1, edge_attr_dim, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels * edge_attr_dim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.edge_attr_dim * self.in_channels
        uniform(size, self.weight)
        uniform(size, self.att_weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.mm(x, self.weight)
        x = x.view(-1, self.edge_attr_dim, self.out_channels)

        row, col = edge_index

        # Compute attention coefficients
        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = (alpha * self.att_weight).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        # This will broadcast edge_attr across all attentions
        alpha = torch.mul(alpha, edge_attr.float())
        alpha = F.normalize(alpha, p=1, dim=1)

        # Sample attention coefficients stochastically.
        dropout = self.dropout if self.training else 0
        alpha = F.dropout(alpha, p=dropout, training=True)

        # Sum up neighborhoods.
        out = alpha.view(-1, self.edge_attr_dim, 1) * x[col]
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        if self.concat is True:
            out = out.view(-1, self.out_channels * self.edge_attr_dim)
        else:
            out = out.sum(dim=1) / self.edge_attr_dim

        if self.bias is not None:
            out = out + self.bias

        return out, alpha

    def __repr__(self):
        return '{}({}, {}, edge_attr_dim={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.edge_attr_dim
                                                     )


class DIFFPool(torch.nn.Module):
    """
    Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper.

    Args:

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_node_features=1):
        super(DIFFPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_features = num_node_features
        self.s = nn.Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.s)

    def forward(self, x, adj):
        """
        Returns pooled node feature matrix, coarsened adjacency matrix and the
        auxiliary link prediction objective
        Args:
            adj: Adjacency matrix with shape [num_edge_features, num_nodes, num_nodes]
        """

        out_x, out_adj, reg = dense_diff_pool(x, adj, self.s)
        out_adj = out_adj.squeeze(0) if out_adj.dim() == 4 else out_adj
        out_edge_index, out_edge_attr = adj_to_edge_index(out_adj)

        return out_x, out_edge_index, out_edge_attr, out_adj, reg

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
