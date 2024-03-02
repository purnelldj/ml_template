import torch
from torchmetrics.classification import BinaryF1Score

from models.base import BaseModel


class UNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f1fun = BinaryF1Score()

    def logits_to_yhat(self, logits: torch.tensor) -> torch.tensor:
        return logits

    def end_of_epoch_metrics(
        self,
        logits: list[torch.tensor],
        y: list[torch.tensor],
        yhat: list[torch.tensor],
        label: str,
        wandb_logger: bool = True,
    ) -> dict:
        yhat = torch.cat(self.yhat_all)
        y = torch.cat(self.y_all)
        f1 = self.f1fun(yhat, y)
        metrics = {}
        metrics[f"{label}_fl"] = f1
        if label == "test":
            acc = self.accuracy(yhat, y)
            logits = torch.cat(self.logits_all)
            loss = self.criterion(logits, y)
            metrics[f"{label}_acc"] = acc
            metrics[f"{label}_loss"] = loss
        self.logger.log_metrics(metrics)

    def on_validation_epoch_end(self) -> None:
        self.end_of_epoch_metrics(self.logits_all, self.y_all, self.yhat_all, "val")

    def on_test_epoch_end(self) -> None:
        self.end_of_epoch_metrics(self.logits_all, self.y_all, self.yhat_all, "test")

    def on_train_epoch_end(self) -> None:
        if self.epoch_counter % 10 == 0:
            self.end_of_epoch_metrics(
                self.logits_all, self.y_all, self.yhat_all, "train"
            )
