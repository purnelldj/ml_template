import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassF1Score

import wandb
from datamodules.EuroSAT_RGB_Samples_utils import class_list
from models.base import BaseModel


class CNN2layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(588, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class CNN2layerBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(588, 10)
        self.drop1 = nn.Dropout(0.7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class CNN3layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(216, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class CNN4layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(48)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class CNN3layerLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(1568, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class CNN4layerLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(576, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class CNN4layerLarge2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=7, stride=1, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding="same"
        )
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding="same"
        )
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class CNNModule(BaseModel):
    def __init__(self, wandb_plots: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.f1fun = MulticlassF1Score(num_classes=10)
        self.wandb_plots = wandb_plots

    def logits_to_yhat(self, logits: torch.tensor) -> torch.tensor:
        return torch.argmax(logits, dim=1)

    def count_classes(self, y: torch.tensor, label: str) -> dict:
        """Count num el in each class."""
        counter = {}
        for idx, classn in enumerate(class_list()):
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
                class_names=class_list(),
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


if __name__ == "__main__":
    net = CNN4layerLarge2()
    tt = torch.rand(1, 3, 64, 64)
    logits = net(tt)
    print(logits.shape)
    pass
