import torch
import torch.nn.functional as F
from torch import nn


class Baseline(torch.nn.Module):

    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(Baseline, self).__init__()
        self.num_nodes = data.num_nodes
        self.num_features = data.num_features
        self.B = data.y.shape[0]
        # self.conv1d = nn.Conv1d(1, 1, kernel_size=self.num_features, stride=self.num_features)
        self.emb1 = nn.Linear(self.num_features, 1)
        self.fc1 = nn.Linear(int(self.num_nodes / self.B), 12)
        self.drop1 = nn.Dropout(dropout)
        # self.fc2 = nn.Linear(8, 8)
        # self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, data):
        if isinstance(data, list):
            print(len(data), data)
            assert len(data) == 1
            data = data[0]

        x = data.x
        B = data.y.shape[0]
        # x = x.view(B, 1, -1)
        # x = self.conv1d(x)
        x = self.emb1(x)
        x = x.view(B, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        # x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        reg = torch.tensor([0], dtype=torch.float, device=x.device)
        return x, reg
