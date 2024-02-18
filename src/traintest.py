import logging

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from datamodules.base import BaseDM
from models.base import BaseModel
from utils import save_hydra_config_to_wandb

# import pprint


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # config logging
    log = logging.getLogger(__name__)
    log.info(f"starting log for: {cfg.group_name}")
    if cfg.log_to_wandb:
        instantiate(cfg.wandb_init)
        save_hydra_config_to_wandb(cfg)

    # get datamodule
    DM: BaseDM = instantiate(cfg.datamodule_inst)
    log.info("successfully instantiated the datamodule")

    # get model: either instantiate or load saved model
    Model: BaseModel = instantiate(cfg.model_inst)
    log.info(f"model hparams: \n {Model.hparams}")

    # visualize datamodule and model output=
    if cfg.visualize_data_and_model:
        visualize_data_model_fun(DM, Model)

    # train model
    if cfg.stage == "fit":
        DM.setup(stage=cfg.stage)
        log.info("setup datamodule for training")
        trainer = L.Trainer()
        trainer.fit(
            model=Model,
            train_dataloaders=DM.train_dataloader(num_workers=11),
            # val_dataloaders=DM.val_dataloader()
        )


def visualize_data_model_fun(DM: BaseDM, Model: BaseModel) -> None:
    DM.setup(stage="fit")
    train_dataloader = DM.train_dataloader()
    for x, y in train_dataloader:
        xplot, yplot = x[0], y[0]
        # now try run the model forwards
        try:
            with torch.no_grad():
                ypred = Model.forward(x)
        except Exception as e:
            print("issue running model")
            raise e
        ypred.detach()
        ypredplot = ypred[0]
        DM.plot_xy(xplot, yplot, ypredplot)
        break


if __name__ == "__main__":
    main()
