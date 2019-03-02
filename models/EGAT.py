import torch
import torch.nn.functional as F
from torch import nn
from nn import EGATConv, DIFFPool


class EGATDP(torch.nn.Module):
    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(EGATDP, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.edge_attr_dim = data.edge_attr.shape[-1]

        self.conv1 = EGATConv(self.num_features, 16,
                              dropout=self.dropout,
                              edge_attr_dim=self.edge_attr_dim
                              )
        self.conv2 = EGATConv(16 * self.edge_attr_dim, 4,
                              dropout=self.dropout,
                              edge_attr_dim=self.edge_attr_dim
                              )

        self.pool1 = DIFFPool(self.num_nodes, 16, self.edge_attr_dim)
        self.pool2 = DIFFPool(16, 1, self.edge_attr_dim)

        self.fc1 = nn.Linear(1 * 4 * self.edge_attr_dim, 20)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(20, 16)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(16, 20)
        self.drop3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x, edge_index, edge_attr, adj, *args, **kwargs):
        """

        :param x: Node feature matrix with shape [num_nodes, num_node_features]
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        :param edge_attr: Edge feature matrix with shape [num_edges, num_edge_features], num_edge_features >= 1
        :param adj: Adjacency matrix with shape [num_edge_features, num_nodes, num_nodes]  # TODO: separately pool multi-dimension adj
        :param args:
        :param kwargs:
        :return:
        """
        if x.dim() == 3:  # a batch
            if x.shape[0] != 1:
                raise Exception("batch size greater than 1 is not supported.")
            x, edge_index, edge_attr, adj = x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0), adj.squeeze(0)
        # # Add self-loops to adjacency matrix.
        # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        # edge_index, edge_attr = add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes=x.size(0))
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
        x, edge_index, e, adj, reg2 = self.pool2(x, adj)
        x = x.view(1, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = F.elu(self.drop3(self.fc3(x)))
        x = self.fc4(x)
        
        reg = reg1 + reg2
        return x, reg
