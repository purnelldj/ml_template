import torch.nn.functional as F
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


class ViTv2(MultiClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            ignore_mismatched_sizes=True,
        )
        self.net.eval()
        self.net.requires_grad_(False)
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x).logits
        logits = F.relu(self.fc1(logits))
        logits = self.fc2(logits)
        return logits


class ViTv3(MultiClass):
    def __init__(self, num_classes: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            ignore_mismatched_sizes=True,
        )
        in_feats = self.net.classifier.in_features
        self.net.classifier = nn.Identity()
        self.net.eval()
        self.net.requires_grad_(False)
        self.drop_out = nn.Dropout(0.7)
        self.fc1 = nn.Linear(in_feats, in_feats)
        self.fc2 = nn.Linear(in_feats, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x).logits
        logits = self.drop_out(logits)
        logits = F.relu(self.fc1(logits))
        logits = self.fc2(logits)
        return logits
