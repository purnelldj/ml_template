import glob
from pathlib import Path

import numpy as np
import rioxarray as rxr
import torch
from PIL import Image
from torch import Tensor
from transformers import ViTImageProcessor

from utils import im_resize


def list_files(dir: str) -> list[str]:
    """List images in eurosat dir."""
    subdirs = glob.glob(dir + "*")
    assert len(subdirs) == 10
    files = []
    suff = ".jpg"
    for subd in subdirs:
        files += glob.glob(subd + "/*" + suff)
        if len(files) == 0:
            suff = ".tif"
            files += glob.glob(subd + "/*" + suff)
    assert len(files) == 10 * 500
    return files


def class_list() -> list[str]:
    all_classes = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]
    return all_classes


def file_to_im(file: str, rgb_or_ms: str = "rgb") -> np.ndarray:
    """Transform jpeg to to numpy."""
    if rgb_or_ms == "rgb":
        return np.array(Image.open(file)).astype(np.float32)
    elif rgb_or_ms == "ms":
        return tif_to_im(file)


def tif_to_im(file: str) -> Tensor:
    xr = rxr.open_rasterio(file)
    arr = xr.values
    return np.transpose(arr, (1, 2, 0))


def file_to_class(file: str) -> str:
    """Extract class from image file path."""
    classp = Path(file)
    classn = classp.name.split("_")[0]
    return classn


def file_to_class_ind(file: str) -> torch.tensor:
    """Extract class number as tensor from image file path."""
    classn = file_to_class(file)
    all_classes = class_list()
    try:
        ind = all_classes.index(classn)
    except ValueError:
        raise Exception("input with unknown class")
    return torch.tensor(ind)


class EuTransform:
    def __init__(
        self,
        im_height: int = 224,
        im_width: int = 224,
        im_channels: int = 3,
        rgb_or_ms: str = "rgb",
    ) -> None:
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = im_channels
        self.rgb_or_ms = rgb_or_ms
        if rgb_or_ms == "rgb":
            self.band_scale = 255.0
        elif rgb_or_ms == "ms":
            self.band_scale = 10000.0

    def __call__(self, file: str) -> None:
        im = file_to_im(file, self.rgb_or_ms)
        im_height, im_width = self.im_height, self.im_width
        if im.shape[0] != im_height or im.shape[1] != im_width:
            im = im_resize(im, im_height, im_width)
        im = np.float16(im)
        im /= self.band_scale
        im = np.transpose(im, (2, 0, 1))
        im = torch.tensor(im)
        label = file_to_class_ind(file)
        return im, label


class ViTransform:
    def __init__(self) -> None:
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

    def __call__(self, file: str) -> None:
        im = file_to_im(file)
        im = self.processor(images=im, return_tensors="pt")
        im = im["pixel_values"][0]
        label = file_to_class_ind(file)
        return im, label
