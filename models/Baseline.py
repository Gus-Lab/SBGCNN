import torch
import torch.nn.functional as F
from torch import nn


class Baseline(torch.nn.Module):

    def __init__(self,
                 data,
                 dropout=0
                 ):
        super(Baseline, self).__init__()
        self.in_channels = data.num_features * data.num_nodes
        self.fc1 = nn.Linear(self.in_channels, 8)
        self.drop1 = nn.Dropout(dropout)
        # self.fc2 = nn.Linear(8, 8)
        # self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x, *args, **kwargs):
        x = x.view(1, -1)
        x = F.elu(self.drop1(self.fc1(x)))
        # x = F.elu(self.drop2(self.fc2(x)))
        x = self.fc3(x)

        reg = torch.tensor([0.0], device=x.device)
        return x, reg
