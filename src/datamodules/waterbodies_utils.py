import albumentations as A
import numpy as np
import torch
from PIL import Image


class WbTransform:
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(self, im: str, mask: str) -> tuple[torch.float32, torch.float32]:
        im = Image.open(im)
        mask = Image.open(mask).convert("L")

        im, mask = np.array(im), np.array(mask)
        transform = A.Compose(
            [
                A.Resize(self.height, self.width),
                # A.HorizontalFlip(),
            ]
        )
        transformed = transform(image=im, mask=mask)
        im = transformed["image"]
        mask = transformed["mask"]

        im = np.transpose(im, (2, 0, 1))
        im = im / 255.0

        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        im = im.astype(np.float32)
        mask = mask.astype(np.float32)

        im = torch.tensor(im)
        mask = torch.tensor(mask)

        mask = torch.round(mask)  # some values that arent 1 or 0

        return im, mask
