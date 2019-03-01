import torch
import torch.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from modules import EGATConv, DIFFPool
from torch_geometric.utils import remove_self_loops
from utils import add_self_loops_with_edge_attr



class Baseline(torch.nn.Module):

    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(data.num_features * data.num_nodes, 24)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(24, 8)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(8, 2)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x = data.x
        x = x.view(-1).unsqueeze(0)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return self.act(x)


class GAT(torch.nn.Module):
    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.conv1 = GATConv(self.num_features, 16, heads=8, dropout=self.dropout)
        self.conv2 = GATConv(16 * 8, 8, dropout=self.dropout)
        self.fc1 = nn.Linear(data.num_nodes * 8, 84)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(84, 24)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(24, 2)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, data):
        """
        Note:
            only one graph in a batch is supported currently.
        """
        x, edge_index = data.x, data.edge_index
        x, edge_index = x.squeeze(0), edge_index.squeeze(0)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = x.view(-1).unsqueeze(0)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return self.act(x)


class EGAT_with_DIFFPool(torch.nn.Module):
    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(EGAT_with_DIFFPool, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.edge_attr_dim1 = data.edge_attr.shape[-1]
        self.edge_attr_dim2 = 4 * self.edge_attr_dim1
        self.edge_attr_dim3 = 1 * self.edge_attr_dim2

        self.conv1 = EGATConv(self.num_features, 8, heads=4,
                              dropout=self.dropout,
                              edge_attr_dim=self.edge_attr_dim1
                              )
        self.conv2 = EGATConv(8 * self.edge_attr_dim2, 8, heads=1,
                              dropout=self.dropout,
                              edge_attr_dim=self.edge_attr_dim2
                              )

        self.pool1 = DIFFPool(self.num_nodes, 16, self.edge_attr_dim3)
        self.pool2 = DIFFPool(16, 1, self.edge_attr_dim3)

        self.fc1 = nn.Linear(1 * 8 * self.edge_attr_dim3, 128)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(32, 16)
        self.drop3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(16, 2)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, data):
        """
        Args:
            data:
                x: Node feature matrix with shape [num_nodes, num_node_features]
                edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
                edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
                adj: Adjacency matrix with shape [num_edge_features, num_nodes, num_nodes]
        Note:
            only one graph in a batch is supported currently.
        """
        x, edge_index, edge_attr, adj = data.x, data.edge_index, data.edge_attr, data.adj
        x, edge_index, edge_attr = x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0)
        # Add self-loops to adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes=x.size(0))
        #         # Doubly stocahstic normalization of edge_attr
        #         edge_attr = doubly_stochastic_normlization(edge_index, edge_attr, num_nodes=x.size(0))
        """      
        NOTE: 
                edge_attr is already normlized and write to file at preprocessing step,
                because doubly_stochastic_normlization() is incredibly slow.
        """
        x, e = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x, e = self.conv2(x, edge_index, e)
        x = F.elu(x)
        x, edge_index, e, adj, reg1 = self.pool1(x, adj)
        x, edge_index, e, adj, reg2 = self.pool2(x, aji)
        x = x.view(-1).unsqueeze(0)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = F.elu(self.drop3(self.fc3(x)))
        x = self.fc4(x)
        return self.act(x), reg1, reg2
