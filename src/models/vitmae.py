from torch import Tensor
from transformers import ViTForImageClassification

from models.base import MultiClass


class ViTMAEmodule(MultiClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # note this image_processor doesnt work here
        # it may be that preprocessing needs to be done in datamodule
        # self.image_processor = AutoImageProcessor.from_pretrained(
        #     "google/vit-base-patch16-224", do_rescale=False
        # )
        self.net = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=10,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x).logits
        return logits
