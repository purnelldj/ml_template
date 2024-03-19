import logging
import os
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import wandb

log = logging.getLogger(__name__)


def instantaite_model_from_ckpt(cfg: DictConfig, ckpt_path: str = None):
    if ckpt_path is None:
        Model = instantiate(cfg)
    else:
        cfg_ckpt = cfg
        cfg_ckpt["_target_"] = cfg._target_ + ".load_from_checkpoint"
        Model = instantiate(cfg_ckpt)
    return Model



def im_resize(im: np.ndarray, im_height: int = 224, im_width: int = 224) -> np.ndarray:
    transform = A.Compose(
        [
            A.Resize(im_height, im_width),
        ]
    )
    transformed = transform(image=im)
    im = transformed["image"]
    return im


def save_hydra_config_to_wandb(cfg: DictConfig):
    log.info(
        f"Hydra config will be saved to WandB as hydra_config.yaml and in wandb run_dir: {wandb.run.dir}"
    )
    # files in wandb.run.dir folder get directly uploaded to wandb
    with open(os.path.join(wandb.run.dir, "hydra_config.yaml"), "w") as fp:
        OmegaConf.save(cfg, f=fp.name, resolve=True)
    wandb.save(os.path.join(wandb.run.dir, "hydra_config.yaml"))


def save_fig(figName, **kwargs):
    figOutDir = ""
    if "figOutDir" in kwargs:
        figOutDir = kwargs.get("figOutDir")
    figName = figOutDir + figName
    plt.savefig(figName, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    # plt.savefig(figName, format='png', dpi=300)
    plt.close()


def check_dir(pathstr: str):
    Path(pathstr).mkdir(parents=True, exist_ok=True)
