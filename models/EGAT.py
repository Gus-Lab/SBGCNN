import torch
import torch.nn.functional as F
from torch import nn
from nn import MEGATConv, EGATConv


class _EGATConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 num_nodes,
                 batch_size
                 ):
        super(_EGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.B = batch_size

        self.conv1 = EGATConv(in_channels, out_channels, dropout=dropout)
        # self.bn1 = nn.BatchNorm1d(self.num_nodes * 4)
        # self.conv2 = EGATConv(4, out_channels, dropout=dropout)
        # self.bn2 = nn.BatchNorm1d(self.num_nodes * 4)
        # self.conv3 = EGATConv(4, out_channels, dropout=dropout)
        # self.bn3 = nn.BatchNorm1d(self.num_nodes * 4)

    def forward(self, x, edge_index, edge_attr):
        x, edge_index, e = self.conv1(x, edge_index, edge_attr)
        # x = self.bn1(x.view(self.B, -1)).view(self.num_nodes * self.B, -1)
        # x, edge_index, e = self.conv2(x, edge_index, e)
        # x = self.bn2(x.view(self.B, -1)).view(self.num_nodes * self.B, -1)
        # x, edge_index, e = self.conv3(x, edge_index, e)
        # x = self.bn3(x.view(self.B, -1)).view(self.num_nodes * self.B, -1)

        return x


class EGAT(torch.nn.Module):
    def __init__(self, data, dropout=0):
        super(EGAT, self).__init__()
        self.num_features = data.num_features
        self.edge_attr_dim = data.edge_attr.shape[-1]
        self.B = data.y.shape[0]
        self.num_nodes = int(data.num_nodes / self.B)

        self.emb1 = nn.Linear(self.num_features - 7, 1)  # for adj

        self.egatconv_channel1 = _EGATConv(8, 4, dropout, self.num_nodes, self.B)
        self.egatconv_channel2 = _EGATConv(8, 4, dropout, self.num_nodes, self.B)
        self.egatconv_channel3 = _EGATConv(8, 4, dropout, self.num_nodes, self.B)

        self.emb2 = nn.Linear(3, 1)

        self.fd = int(4 * self.num_nodes)

        self.bn1 = nn.BatchNorm1d(self.fd)

        self.fc1 = nn.Linear(self.fd, 4)
        self.drop1 = nn.Dropout(dropout)
        # self.bn2 = nn.BatchNorm1d(6)
        # self.fc2 = nn.Linear(12, 8)
        # self.drop2 = nn.Dropout(dropout)
        # self.bn3 = nn.BatchNorm1d(8)
        # self.fc3 = nn.Linear(8, 6)
        # self.drop3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(4, 2)

    def forward(self, x, edge_index, edge_attr, y):
        if x.dim() == 3:
            x, edge_index, edge_attr, y = \
                x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0), y.squeeze(0)

        B = y.shape[0]

        x = torch.cat([self.emb1(x[:, 7:]), x[:, :7]], dim=-1)

        x = torch.stack([
            self.egatconv_channel1(x, edge_index, edge_attr[:, 0].view(-1, 1)),
            self.egatconv_channel2(x, edge_index, edge_attr[:, 1].view(-1, 1)),
            self.egatconv_channel3(x, edge_index, edge_attr[:, 2].view(-1, 1))
        ], dim=-1)
        x = self.emb2(x)
        x = x.view(B, -1)
        x = self.drop1(self.bn1(F.relu(self.fc1(x))))

        # x = F.elu(self.drop2(self.fc2(x)))
        # x = F.elu(self.drop3(self.fc3(self.bn3(x))))
        x = self.fc4(x)

        reg = torch.tensor([0], dtype=torch.float, device=x.device)
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
