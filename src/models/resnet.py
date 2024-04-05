import torch
import torch.nn.functional as F

# from torchvision import models
from torch import Tensor, nn
from torch.nn import Identity

from models.base import MultiClass


class ResNet(MultiClass):
    def __init__(
        self,
        resnet_size: int = 18,
        pretrained: bool = True,
        num_classes: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # sizes avail: 18, 34, 50, 101, 152
        netstr = "resnet" + str(resnet_size)
        model = torch.hub.load("pytorch/vision:v0.10.0", netstr, pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        self.net = model


class ResNetFrozen(MultiClass):
    def __init__(
        self,
        resnet_size: int = 18,
        pretrained: bool = True,
        num_classes: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # sizes avail: 18, 34, 50, 101, 152
        netstr = "resnet" + str(resnet_size)
        model = torch.hub.load("pytorch/vision:v0.10.0", netstr, pretrained=pretrained)
        model.eval()
        model.requires_grad_(False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_features)
        self.net = model
        self.drop_out = nn.Dropout(0.7)
        self.fc_out = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        logits = F.relu(self.net(x))
        logits = self.drop_out(logits)
        logits = self.fc_out(logits)
        return logits


class ResNetFrozen2(MultiClass):
    def __init__(
        self,
        resnet_size: int = 18,
        pretrained: bool = True,
        num_classes: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # sizes avail: 18, 34, 50, 101, 152
        netstr = "resnet" + str(resnet_size)
        model = torch.hub.load("pytorch/vision:v0.10.0", netstr, pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = Identity()
        model.eval()
        model.requires_grad_(False)
        self.net = model
        self.fc1 = nn.Linear(num_features, num_features)
        self.drop_out = nn.Dropout(0.7)
        self.fc2 = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x)
        logits = F.relu(self.fc1(logits))
        logits = self.drop_out(logits)
        logits = self.fc2(logits)
        return logits


class ResNet13chanV1(MultiClass):
    def __init__(
        self,
        resnet_size: int = 18,
        pretrained: bool = True,
        num_classes: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # sizes avail: 18, 34, 50, 101, 152
        netstr = "resnet" + str(resnet_size)
        model = torch.hub.load("pytorch/vision:v0.10.0", netstr, pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = Identity()
        model.eval()
        model.requires_grad_(False)
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=3, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.net = model
        self.drop_out = nn.Dropout(0.7)
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        logits = F.relu(self.bn1(self.conv1(x)))
        logits = self.net(logits)
        logits = self.drop_out(logits)
        logits = F.relu(self.fc1(logits))
        logits = self.fc2(logits)
        return logits
