import glob

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split

from datamodules.waterbodies_utils import im_mask_transform


class WBDS(Dataset):
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.dir_ims = cfg.dirs.ims
        self.dir_masks = cfg.dirs.masks
        self.height = cfg.dims.height
        self.width = cfg.dims.width

        # first get all files
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


class WBDM(pl.LightningDataModule):
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule-api

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # inherit from class
        self.cfg = cfg
        self.test_size = cfg.test_size
        self.val_size = cfg.val_size
        self.seed = cfg.seed
        self.batch_size = cfg.batch_size

    def prepare_data(self):
        # for downloading and tokenizing data
        pass

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

        print("train / val / test split: ")
        print(f"{len(self.wb_train)} / {len(self.wb_val)} / {len(self.wb_test)}")

    def train_dataloader(self):
        return DataLoader(self.wb_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.wb_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.wb_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        # this is for unlabeled data
        pass
