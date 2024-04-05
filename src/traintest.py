import cProfile
import logging
import pstats

import hydra
import lightning as L
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.logger import Logger
from omegaconf import DictConfig

import wandb
from datamodules.base import BaseDM
from models.base import BaseModel
from utils import instantaite_model_from_ckpt, save_fig, save_hydra_config_to_wandb


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
    datamodule: BaseDM = instantiate(cfg.dataset)
    log.info("successfully instantiated the datamodule")

    # visualize datamodule output
    if cfg.visualize_data:
        visualize_data_model_fun(datamodule)
        return

    # get model: either instantiate or load saved model
    # Model: BaseModel = instantiate(cfg.model)
    model: BaseModel = instantaite_model_from_ckpt(cfg.model, cfg.ckpt_path)
    log.info(f"model hparams: \n {model.hparams}")
    log.info(model)

    log.info("instantiating trainer")
    trainer: Trainer = instantiate(cfg.trainer, logger=logger)

    # test to see if model accepts input
    test_data_shape(datamodule, model, trainer)
    log.info("TEST PASSED: model accepts data shape as input")

    # visualize datamodule and model outputs
    if cfg.visualize_modelout:
        visualize_data_model_fun(datamodule, model, trainer)
        return

    if cfg.stage == "fit":
        log.info("training model...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.stage == "test":
        log.info("testing model...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.logger_name == "wandb":
        wandb.finish()


def test_data_shape(
    datamodule: BaseDM, model: BaseModel = None, trainer: Trainer = None
) -> None:
    """Test if output from dataloader is compatible with model."""
    datamodule.setup()
    subset = torch.utils.data.Subset(datamodule.xy_train, [0])
    single_dl = torch.utils.data.DataLoader(subset, batch_size=1)
    try:
        trainer.predict(model, dataloaders=single_dl)
    except Exception as e:
        print(e)
        x, _ = next(iter(single_dl))
        raise Exception(f"issue passing input of shape {x.shape} to model")


def visualize_data_model_fun(
    datamodule: BaseDM, model: BaseModel = None, trainer: Trainer = None, idx: int = 5
) -> None:
    """For checking datamodule and model outputs prior to training."""
    datamodule.setup()
    subset = torch.utils.data.Subset(datamodule.xy_train, [*range(12)])
    single_dl = torch.utils.data.DataLoader(subset, batch_size=12)
    x, y = next(iter(single_dl))
    xplot, yplot = x[idx], y[idx]
    if model is not None:
        predict_out = trainer.predict(model, dataloaders=single_dl)
        logits = predict_out[0]
        print(f"output logits are of type: {type(logits)}")
        print(f"and size: {logits.shape}")
        yhat = model.logits_to_yhat(logits)
        loss = model.criterion(logits, y)
        acc = model.accuracy(yhat, y)
        print(f"testing loss: {loss}")
        print(f"testing accuracy: {acc}")
        ypredplot = yhat[idx]
        datamodule.plot_xy(xplot, yplot, ypredplot)
    else:
        datamodule.plot_xy(xplot, yplot)
    save_fig("data_model_out_tmp.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    profiler = False
    if profiler:
        # if you want to search for bottlenecks:
        with cProfile.Profile() as profile:
            main()
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats(10)
    else:
        main()
