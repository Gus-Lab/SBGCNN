import torch
import torch.nn.functional as F
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

        self.conv1 = EGATConv(in_channels, 8, heads=5, dropout=dropout, concat=True)
        # self.bn1 = nn.BatchNorm1d(40)

        # self.conv2 = EGATConv(40, 40, heads=1, dropout=dropout, concat=False)

        self.pconv1 = EGATConv(40, 32)
        self.pool1 = DIFFPool()

        self.pconv2 = EGATConv(40, 8)
        self.pool2 = DIFFPool()
        # self.bn2 = nn.BatchNorm1d(40)

        self.pconv3 = EGATConv(40, 1)
        self.pool3 = DIFFPool()

    def forward(self, x, edge_index, edge_attr, adj):
        x, edge_index, e = self.conv1(x, edge_index, edge_attr)
        e = self.dot(e).unsqueeze(-1)
        self.writer.add_histogram('conv1_x_std', x.std(dim=0))
        # x = self.bn1(x)

        # x, edge_index, e = self.conv2(x, edge_index, e)
        # e = self.dot(e).unsqueeze(-1)
        # self.writer.add_histogram('conv2_x_std', x.std(dim=0))

        s, _, _ = self.pconv1(x, edge_index, edge_attr)
        x, edge_index, edge_attr, adj, reg1 = self.pool1(x, adj, s)
        self.writer.add_histogram('pool1_x_std', x.std(dim=0))
        print("pool1_x", x[:, 0])

        s, _, _ = self.pconv2(x, edge_index, edge_attr)
        x, edge_index, edge_attr, adj, reg2 = self.pool2(x, adj, s)
        self.writer.add_histogram('pool2_x_std', x.std(dim=0))
        # x = self.bn2(x)
        print("pool2_x", x[:, 0])

        s, _, _ = self.pconv3(x, edge_index, edge_attr)
        x, edge_index, edge_attr, adj, reg3 = self.pool3(x, adj, s)
        print("pool3_x", x)

        # reg = torch.tensor([0], dtype=torch.float, device=x.device)
        reg = reg1 + reg2 + reg3
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

        self.fc1 = nn.Linear(40 * 1, 32)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, edge_index, edge_attr, y, adj):
        if x.dim() == 3:
            x, edge_index, edge_attr, y = \
                x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0), y.squeeze(0)

        if adj.dim() == 3:
            adj = adj.squeeze(0)

        x, reg = self.egatconv_channel1(x, edge_index, edge_attr, adj)
        x = x.view(self.B, -1)
        x = self.drop1(F.relu(self.fc1(x)))
        # x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc2(x)

        # reg = torch.tensor([0], dtype=torch.float, device=x.device)
        return x, reg


class MEGAT(torch.nn.Module):
    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(MEGAT, self).__init__()
        self.dropout = dropout
        self.num_features = data.num_features
        self.num_nodes = data.num_nodes
        self.edge_attr_dim = data.edge_attr.shape[-1]

        self.conv1 = MEGATConv(self.num_features, 16,
                               dropout=self.dropout,
                               edge_attr_dim=self.edge_attr_dim
                               )
        self.conv2 = MEGATConv(16 * self.edge_attr_dim, 4,
                               dropout=self.dropout,
                               edge_attr_dim=self.edge_attr_dim
                               )

        self.fc1 = nn.Linear(self.num_nodes * 4 * self.edge_attr_dim, 20)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(20, 16)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(16, 20)
        self.drop3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, data):
        """

        :param x: Node feature matrix with shape [num_nodes, num_node_features]
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        :param edge_attr: Edge feature matrix with shape [num_edges, num_edge_features], num_edge_features >= 1
        :param adj: Adjacency matrix with shape [num_edge_features, num_nodes, num_nodes]  # TODO: separately pool multi-dimension adj
        :param args:
        :param kwargs:
        :return:
        """
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
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
        B = data.y.shape[0]
        x, e = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x, e = self.conv2(x, edge_index, e)
        x = F.elu(x)
        x = x.view(B, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = F.elu(self.drop3(self.fc3(x)))
        x = self.fc4(x)

        reg = torch.tensor([0], dtype=torch.float, device=x.device)
        return x, reg
