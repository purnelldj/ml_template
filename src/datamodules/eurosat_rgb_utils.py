import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils import im_resize


def list_files(dir: str) -> list[str]:
    """List images in 'eurosat_rgb' dir."""
    subdirs = glob.glob(dir + "*")
    assert len(subdirs) == 10
    files = []
    for subd in subdirs:
        files += glob.glob(subd + "/*.jpg")
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


def file_to_im(
    file: str, im_height: int = 224, im_width: int = 224, im_channels: int = 3, **kwargs
) -> torch.tensor:
    """Transform jpeg to Tensor."""
    im = np.array(Image.open(file)).astype(np.float32)
    if im.shape[0] != im_height or im.shape[1] != im_width:
        im = im_resize(im, im_height, im_width)
    assert im.shape == (im_height, im_width, im_channels)
    im /= 255.0
    im = np.transpose(im, (2, 0, 1))
    return torch.tensor(im)


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
