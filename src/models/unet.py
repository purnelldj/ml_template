import segmentation_models_pytorch as smp
from omegaconf import DictConfig
from segmentation_models_pytorch.losses import DiceLoss

from models.templatemodel import TemplateModel


class UNet(TemplateModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.net = smp.Unet(
            encoder_name=cfg.ENCODER,
            encoder_weights=cfg.WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.criterion = DiceLoss(mode="binary")

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
