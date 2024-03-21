import torch

# from torchvision import models
from torch import nn

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
