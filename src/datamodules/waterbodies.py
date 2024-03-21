import glob
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from datamodules.base import BaseDM, BaseDS

log = logging.getLogger(__name__)


class WBDS(BaseDS):
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    def __init__(self, dir_ims: str, dir_masks: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dir_ims = dir_ims
        self.dir_masks = dir_masks

        # all files
        ims_all = glob.glob(self.dir_ims + "water_body_*")
        masks_all = glob.glob(self.dir_masks + "water_body_*")
        self.ims_all = np.sort(np.array(ims_all))
        self.masks_all = np.sort(np.array(masks_all))
        assert len(ims_all) == len(masks_all)

    def __len__(self) -> int:
        return len(self.ims_all)

    def __getitem__(self, idx):
        return self.transform(self.ims_all[idx], self.masks_all[idx])


class WBDM(BaseDM):
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule-api

    def __init__(self, mask_ratio: int = 0.5, **kwargs):
        super().__init__(**kwargs)
        # inherit from class
        self.mask_ratio = mask_ratio

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
        axarr[0].set_title("image")
        axarr[1].imshow(ynp, cmap="gray")
        axarr[1].set_title("mask")
        if ypred is not None:
            ypred = torch.sigmoid(ypred)
            ypred = (ypred > self.mask_ratio) * 1.0
            yprednp = np.squeeze(ypred.numpy())
            axarr[2].imshow(yprednp, cmap="gray")
            axarr[2].set_title("prediction")
        for ax in axarr:
            ax.axis("off")
