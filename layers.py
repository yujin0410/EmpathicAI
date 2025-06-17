import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, drop):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        return x
