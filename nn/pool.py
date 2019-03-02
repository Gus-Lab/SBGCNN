import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import dense_diff_pool
from utils import adj_to_edge_index
from torch_geometric.nn.inits import uniform, glorot
from torch_geometric.nn import global_sort_pool


class SortPool(torch.nn.Module):

    def __init__(self, k):
        super(SortPool, self).__init__()
        self.k = k

    def forward(self, x):
        return global_sort_pool(x=x,
                                batch=torch.tensor([0 for i in range(x.size()[0])], dtype=torch.long, device=x.device),
                                k=self.k)

    def __repr__(self):
        return '{}(k_nodes_to_keep={})'.format(self.__class__.__name__,
                                               self.k,
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
        # TODO: separately pool multi-dimension adj
        out_x, out_adj, reg = dense_diff_pool(x, adj, self.s)
        out_adj = out_adj.squeeze(0) if out_adj.dim() == 4 else out_adj
        out_edge_index, out_edge_attr = adj_to_edge_index(out_adj)

        return out_x, out_edge_index, out_edge_attr, out_adj, reg

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
