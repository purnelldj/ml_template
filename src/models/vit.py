from torch import Tensor, nn
from transformers import ViTForImageClassification

from models.base import MultiClass


class ViTmodule(MultiClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=10,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x).logits
        return logits


class ViTmodulev2(MultiClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            ignore_mismatched_sizes=True,
        )
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)


    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x).logits
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        return logits
