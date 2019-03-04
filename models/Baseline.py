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
        self.conv1d = nn.Conv1d(1, 1, kernel_size=7, stride=7)
        self.fc1 = nn.Linear(129, 6)
        self.drop1 = nn.Dropout(dropout)
        # self.fc2 = nn.Linear(8, 8)
        # self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(6, 2)

    def forward(self, x, *args, **kwargs):
        x = x.view(1, 1, -1)
        x = self.conv1d(x)
        x = x.view(1, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        # x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        reg = torch.tensor([0.0], device=x.device)
        return x, reg
