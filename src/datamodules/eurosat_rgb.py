import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from datamodules.base import BaseDM, BaseDS
from datamodules.eurosat_rgb_utils import class_list, file_to_class, list_files


class EUsatrgbDS(BaseDS):
    def __init__(self, dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ims_all = np.array(list_files(dir))
        self.labels_all = [file_to_class(file) for file in self.ims_all]
        self.labels_all = np.array(self.labels_all)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.ims_all)

    def __getitem__(self, idx):
        """Return X, y for given id."""
        return self.transform(self.ims_all[idx])


class EUsatrgbDM(BaseDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # inherit from class
        for i, class_name in enumerate(class_list()):
            self.label2id[class_name] = i
            self.id2label[i] = class_name

    def setup(self, stage: str = "fit"):
        """Perform train/val/test splits, create datasets, apply transforms."""
        # Assign Train/val split(s) for use in Dataloaders
        DS = self.Dataset
        indices = range(len(DS))
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=DS.labels_all,
        )
        val_size_adj = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_size_adj,
            random_state=self.seed,
            stratify=DS.labels_all[trainval_idx],
        )
        train_idx = np.array(train_idx)
        self.xy_train = Subset(DS, train_idx)
        self.xy_val = Subset(DS, val_idx)
        self.xy_test = Subset(DS, test_idx)

        self.log.info("train / val / test split: ")
        self.log.info(
            f"{len(self.xy_train)} / {len(self.xy_val)} / {len(self.xy_test)}"
        )

    def plot_xy(
        self, x: torch.tensor, y: torch.tensor, ypred: torch.tensor = None
    ) -> None:
        """To plot x, y and prediction."""
        class_true = class_list()[y.numpy()]
        _, ax = plt.subplots()
        xnmp = x.numpy()
        xnmp = np.transpose(xnmp, (1, 2, 0))
        ax.imshow(xnmp)
        title = f"true class = {class_true}"
        if ypred is not None:
            class_pred = class_list()[np.argmax(ypred.numpy())]
            title += f", pred class = {class_pred}"
        ax.set_title(title)
        ax.axis("off")
