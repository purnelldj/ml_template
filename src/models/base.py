from typing import Tuple

import lightning as L
import torch

from datamodules.base import BaseDM

"""
https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightningmodule
A LightningModule organizes your PyTorch code into 6 sections:
Initialization (__init__ and setup()).
Train Loop (training_step())
Validation Loop (validation_step())
Test Loop (test_step())
Prediction Loop (predict_step())
Optimizers and LR Schedulers (configure_optimizers())

also taken inspiration from https://github.com/ashleve/lightning-hydra-template
"""


class BaseModel(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        accuracy,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        DM: BaseDM = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.accuracy = accuracy

        self.net = net
        self.criterion = criterion

        self.epoch_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        x, y = batch
        loss = self.trainval_log_step(x, y, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.trainval_log_step(x, y, "val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        yhat = self.logits_to_yhat(logits)

        self.logits_all.append(logits)
        self.y_all.append(y)
        self.yhat_all.append(yhat)

    def trainval_log_step(self, x: torch.tensor, y: torch.tensor, train_or_val: str):
        logits = self(x)
        yhat = self.logits_to_yhat(logits)
        loss = self.criterion(logits, y)
        acc = self.accuracy(yhat, y)

        self.y_all.append(y)
        self.yhat_all.append(yhat)

        # update and log metrics
        self.log(
            train_or_val + "_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            train_or_val + "_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if train_or_val == "train":
            return loss

    def on_train_epoch_start(self) -> None:
        self.epoch_counter += 1
        self.logits_all = []
        self.y_all = []
        self.yhat_all = []

    def on_validation_epoch_start(self) -> None:
        self.logits_all = []
        self.y_all = []
        self.yhat_all = []

    def on_test_epoch_start(self) -> None:
        self.logits_all = []
        self.y_all = []
        self.yhat_all = []

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return optimizer

    def logits_to_yhat(self, logits: torch.tensor) -> torch.tensor:
        return logits
