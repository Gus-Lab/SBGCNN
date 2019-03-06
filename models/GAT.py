import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


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

    def forward(self, x, edge_index, *args, **kwargs):
        """
        Note:
            only one graph in a batch is supported.
        """
        if x.dim() == 3:  # a batch
            if x.shape[0] != 1:
                raise Exception("batch size greater than 1 is not supported.")
            x, edge_index = x.squeeze(0), edge_index.squeeze(0)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = x.view(-1).unsqueeze(0)
        x = F.elu(self.drop1(self.fc1(x)))
        x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        reg = torch.tensor([0], dtype=torch.float, device=x.device)
        return x, reg
