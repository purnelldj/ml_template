import cProfile
import logging
import pstats

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers.logger import Logger
from omegaconf import DictConfig

import wandb
from datamodules.base import BaseDM
from models.base import BaseModel
from utils import save_hydra_config_to_wandb


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    if cfg.logger_name == "wandb":
        logp = cfg.logger
        wandb.init(
            name=logp.name, group=logp.group, project=logp.project, mode=logp.mode
        )
        save_hydra_config_to_wandb(cfg)

    # config logging
    log = logging.getLogger(__name__)
    log.info(f"starting log for: {cfg.group_name}")

    logger: Logger = instantiate(cfg.logger)
    log.info("instantiated logger")

    # seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)

    # get datamodule
    DM: BaseDM = instantiate(cfg.datamodule)
    log.info("successfully instantiated the datamodule")

    # visualize datamodule output
    if cfg.visualize_data:
        visualize_data_model_fun(DM)
        return

    # get model: either instantiate or load saved model
    Model: BaseModel = instantiate(cfg.model, DM=DM)
    log.info(f"model hparams: \n {Model.hparams}")
    log.info(Model)

    # visualize datamodule and model outputs
    if cfg.visualize_modelout:
        visualize_data_model_fun(DM, Model)
        return

    log.info("instantiating trainer")
    trainer = instantiate(cfg.trainer, logger=logger)

    if cfg.stage == "fit":
        log.info("training model...")
        trainer.fit(model=Model, datamodule=DM, ckpt_path=cfg.ckpt_path)

    if cfg.stage == "test":
        log.info("testing model...")
        trainer.test(model=Model, datamodule=DM, ckpt_path=cfg.ckpt_path)


def visualize_data_model_fun(DM: BaseDM, Model: BaseModel = None, idx: int = 3) -> None:
    """For checking datamodule and model outputs prior to training."""
    DM.setup()
    train_dataloader = DM.train_dataloader()
    for x, y in train_dataloader:
        xplot, yplot = x[idx], y[idx]
        # now try run the model forwards
        if Model is not None:
            try:
                with torch.no_grad():
                    logits = Model.forward(x)
            except Exception as e:
                print("issue running model")
                raise e
            logits.detach()
            loss = Model.criterion(logits, y)
            acc = Model.accuracy(logits, y)
            print(f"testing loss: {loss}")
            print(f"testing accuracy: {acc}")
            ypredplot = logits[idx]
            DM.plot_xy(xplot, yplot, ypredplot)
        else:
            DM.plot_xy(xplot, yplot)
        break


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        main()
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats(10)
