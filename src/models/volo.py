import timm
from torch import Tensor, nn

from models.base import MultiClass


class Volo(MultiClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = timm.create_model("volo_d1_224.sail_in1k", pretrained=True)
        self.fc1 = nn.Linear(1000, 10)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x)
        logits = self.fc1(logits)
        return logits


class Volov2(MultiClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = timm.create_model("volo_d1_224.sail_in1k", pretrained=True)
        self.net.eval()
        self.net.requires_grad_(False)
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        logits = self.fc3(logits)
        return logits
