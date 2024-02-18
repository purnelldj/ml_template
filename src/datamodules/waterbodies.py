import glob
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split

from datamodules.base import BaseDM
from datamodules.waterbodies_utils import im_mask_transform

log = logging.getLogger(__name__)


class WBDS(Dataset):
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.dir_ims = cfg.dirs.ims
        self.dir_masks = cfg.dirs.masks
        self.height = cfg.dims.height
        self.width = cfg.dims.width

        # all files
        ims_all = glob.glob(self.dir_ims + "water_body_*")
        masks_all = glob.glob(self.dir_masks + "water_body_*")
        self.ims_all = np.array(ims_all)
        self.masks_all = np.array(masks_all)
        assert len(ims_all) == len(masks_all)

    def __len__(self) -> int:
        return len(self.ims_all)

    def __getitem__(self, idx):
        im, mask = im_mask_transform(
            self.ims_all[idx], self.masks_all[idx], self.height, self.width
        )
        return im, mask


class WBDM(BaseDM):
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule-api

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # inherit from class
        self.mask_ratio = cfg.mask_ratio

    def setup(self, stage: str):
        # count number of classes
        # build vocabulary
        # perform train/val/test splits
        # create datasets
        # apply transforms (defined explicitly in your datamodule)

        # Assign Train/val split(s) for use in Dataloaders
        wb_full = WBDS(self.cfg)
        self.wb_train, self.wb_val, self.wb_test = random_split(
            wb_full,
            [1 - self.val_size - self.test_size, self.val_size, self.test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.wb_predict = None

        log.info("train / val / test split: ")
        log.info(f"{len(self.wb_train)} / {len(self.wb_val)} / {len(self.wb_test)}")

    def plot_xy(
        self, x: torch.tensor, y: torch.tensor, ypred: torch.tensor = None
    ) -> None:
        """To plot x, y and prediction."""
        if len(x.shape) > 3:
            raise Exception("plot one image at a time")
        plotl = 2
        if ypred is not None:
            plotl = 3
        _, axarr = plt.subplots(1, plotl)
        xnp = np.transpose(x.numpy(), (1, 2, 0))
        ynp = np.squeeze(y.numpy())
        axarr[0].imshow(xnp)
        axarr[1].imshow(ynp, cmap="gray")
        if ypred is not None:
            ypred = torch.sigmoid(ypred)
            ypred = (ypred > self.mask_ratio) * 1.0
            yprednp = np.squeeze(ypred.numpy())
            axarr[2].imshow(yprednp, cmap="gray")
        plt.show()
        plt.close()
