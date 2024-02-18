import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig
from segmentation_models_pytorch.losses import DiceLoss

from models.base import BaseModel


class UNet(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.net = smp.Unet(
            encoder_name=cfg.ENCODER,
            encoder_weights=cfg.WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.LR = cfg.LR
        self.momentum = cfg.momentum
        self.criterion = DiceLoss(mode="binary")
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=cfg.LR, momentum=0.9)
        self.compile = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.LR, momentum=self.momentum
        )
        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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
