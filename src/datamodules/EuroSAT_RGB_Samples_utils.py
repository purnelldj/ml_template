import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def list_files(dir: str) -> list[str]:
    """List images in 'EuroSAT_RGB_Samples' dir."""
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


def file_to_im(file: str) -> torch.tensor:
    """Transform jpeg to Tensor."""
    im = np.array(Image.open(file)).astype(np.float32)
    assert im.shape == (64, 64, 3)
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
