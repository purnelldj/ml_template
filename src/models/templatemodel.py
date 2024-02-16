from typing import Tuple

import lightning as L
import torch

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


class TemplateModel(L.LightningModule):
    def __init__(
        self,
        # net: torch.nn.Module = None,
        # criterion: torch.nn.modules.loss._Loss = None,
        # optimizer: torch.optim.Optimizer,
        # scheduler: torch.optim.lr_scheduler,
        # compile: bool,
    ) -> None:
        super().__init__()
        self.net = torch.nn.Module()
        self.criterion = torch.nn.modules.loss._Loss()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict."""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # update and log metrics
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # implement your own
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # calculate acc
        val_acc = None

        # log the outputs!
        self.log_dict({"val_loss": loss, "val_acc": val_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch

        # implement your own
        out = self(x)
        loss = self.loss(out, y)

        # calculate acc
        test_acc = None

        # log the outputs!
        self.log_dict({"val_loss": loss, "test_acc": test_acc})

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
