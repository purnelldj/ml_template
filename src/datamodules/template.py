import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split


class TemplateDS(Dataset):
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def __len__(self) -> int:
        """Return length of dataset."""
        pass

    def __getitem__(self, idx):
        """Return X, y for given id."""
        pass


class TemplateDM(pl.LightningDataModule):
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
        """For downloading and tokenizing data."""
        pass

    def setup(self):
        """Perform train/val/test splits, create datasets, apply transforms."""
        # Assign Train/val split(s) for use in Dataloaders
        wb_full = TemplateDS(self.cfg)
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
