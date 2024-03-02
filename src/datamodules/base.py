import logging

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class BaseDS(Dataset):
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    def __init__(self, **kwargs) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        """Return length of dataset."""
        pass

    def __getitem__(self, idx):
        """Return X, y for given id."""
        pass


class BaseDM(L.LightningDataModule):
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule-api

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        seed: int = 1,
        batch_size: int = 12,
        dir: str = None,
        Dataset: Dataset = None,
        **kwargs,
    ):
        super().__init__()
        # inherit from class
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.batch_size = batch_size
        self.dir = dir
        self.log = logging.getLogger(__name__)
        # now instantiate the dataset
        self.Dataset = Dataset

    def prepare_data(self):
        """For downloading and tokenizing data."""
        pass

    def setup(self, stage: str = "fit"):
        """Perform train/val/test splits, create datasets, apply transforms."""
        # Assign Train/val split(s) for use in Dataloaders
        xy_full = self.Dataset
        self.xy_train, self.xy_val, self.xy_test = random_split(
            xy_full,
            [1 - self.val_size - self.test_size, self.val_size, self.test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.xy_predict = None

        self.log.info("train / val / test split: ")
        self.log.info(
            f"{len(self.xy_train)} / {len(self.xy_val)} / {len(self.xy_test)}"
        )

    def train_dataloader(self, **kwargs):
        return DataLoader(self.xy_train, batch_size=self.batch_size, **kwargs)

    def val_dataloader(self, **kwargs):
        return DataLoader(self.xy_val, batch_size=self.batch_size, **kwargs)

    def test_dataloader(self, **kwargs):
        return DataLoader(self.xy_test, batch_size=self.batch_size, **kwargs)

    def predict_dataloader(self):
        # this is for unlabeled data
        pass

    def plot_xy(
        self, x: torch.tensor, y: torch.tensor, ypred: torch.tensor = None
    ) -> None:
        """To plot x, y and prediction."""
        x, y, ypred
        raise Exception("this is a template class")
