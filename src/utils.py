import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

import wandb

log = logging.getLogger(__name__)


def save_hydra_config_to_wandb(cfg: DictConfig):
    log.info(
        f"Hydra config will be saved to WandB as hydra_config.yaml and in wandb run_dir: {wandb.run.dir}"
    )
    # files in wandb.run.dir folder get directly uploaded to wandb
    with open(os.path.join(wandb.run.dir, "hydra_config.yaml"), "w") as fp:
        OmegaConf.save(cfg, f=fp.name, resolve=True)
    wandb.save(os.path.join(wandb.run.dir, "hydra_config.yaml"))


def saveFig(figName, **kwargs):
    figOutDir = ""
    if "figOutDir" in kwargs:
        figOutDir = kwargs.get("figOutDir")
    figName = figOutDir + figName
    plt.savefig(figName, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    # plt.savefig(figName, format='png', dpi=300)
    plt.close()


def check_dir(pathstr: str):
    Path(pathstr).mkdir(parents=True, exist_ok=True)
