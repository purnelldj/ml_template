import torch
import torch.nn.functional as F
from torch import nn


class CNN3layer(nn.Module):
    """Requires 64x64 input."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(216, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x
