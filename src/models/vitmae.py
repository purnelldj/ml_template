import torch
from torch import Tensor, nn
from transformers import AutoImageProcessor, ViTForImageClassification

from datamodules.base import BaseDM
from models.cnn import CNNModule


class ViTMAE(nn.Module):
    def __init__(self, DM: BaseDM):
        super().__init__()
        # https://huggingface.co/docs/transformers/v4.38.2/en/model_doc/vit#transformers.ViTForImageClassification
        self.image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224", do_rescale=False
        )
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            id2label=DM.id2label,
            label2id=DM.label2id,
            num_labels=len(DM.label2id),
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        inputs = self.image_processor(x, return_tensors="pt")
        logits = self.model.forward(**inputs).logits
        return logits


class ViTMAEmodule(CNNModule):
    def __init__(self, DM: BaseDM, **kwargs) -> None:
        super().__init__(**kwargs)
        self.net = ViTMAE(DM)

    def logits_to_yhat(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=1)
