from typing import Tuple

import lightning as L
import torch
from torchmetrics.classification import MulticlassF1Score

import wandb
from datamodules.base import BaseDM

"""
taken inspiration from :
https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightningmodule
https://github.com/ashleve/lightning-hydra-template
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


class MultiClass(BaseModel):
    # base module for multi class problems
    def __init__(
        self, wandb_plots: bool = True, class_list: list[str] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.f1fun = MulticlassF1Score(num_classes=10)
        self.wandb_plots = wandb_plots
        self.class_list = class_list

    def logits_to_yhat(self, logits: torch.tensor) -> torch.tensor:
        return torch.argmax(logits, dim=1)

    def count_classes(self, y: torch.tensor, label: str) -> dict:
        """Count num el in each class."""
        counter = {}
        for idx, classn in enumerate(self.class_list):
            counter[f"{classn}"] = torch.sum(y == idx)
        return counter

    def end_of_epoch_metrics(
        self,
        logits: list[torch.tensor],
        y: list[torch.tensor],
        yhat: list[torch.tensor],
        label: str,
    ) -> dict:
        yhat = torch.cat(self.yhat_all)
        y = torch.cat(self.y_all)
        f1 = self.f1fun(yhat, y)
        counter = self.count_classes(yhat, label)
        metrics = {}
        metrics[f"{label}_fl"] = f1
        if label == "test":
            acc = self.accuracy(yhat, y)
            logits = torch.cat(self.logits_all)
            loss = self.criterion(logits, y)
            metrics[f"{label}_acc"] = acc
            metrics[f"{label}_loss"] = loss
        self.logger.log_metrics(metrics)
        if self.wandb_plots:
            confmat = wandb.plot.confusion_matrix(
                y_true=y,
                preds=yhat,
                # y_true=y.numpy(),
                # preds=yhat.numpy(),
                class_names=self.class_list,
                title=f"{label}_confmat",
            )
            wandb.log({f"{label}_conf_mat": confmat})
            data = [[key, int(counter[key].numpy())] for key in counter]
            table = wandb.Table(data=data, columns=["class", "count"])
            wandb.log(
                {
                    f"{label}_class_counter": wandb.plot.bar(
                        table, "class", "count", title=f"{label}_class_counter"
                    )
                }
            )

    def on_validation_epoch_end(self) -> None:
        self.end_of_epoch_metrics(self.logits_all, self.y_all, self.yhat_all, "val")

    def on_test_epoch_end(self) -> None:
        self.end_of_epoch_metrics(self.logits_all, self.y_all, self.yhat_all, "test")

    def on_train_epoch_end(self) -> None:
        if self.epoch_counter % 10 == 0:
            self.end_of_epoch_metrics(
                self.logits_all, self.y_all, self.yhat_all, "train"
            )
