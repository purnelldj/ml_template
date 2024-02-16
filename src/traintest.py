import logging
import pprint

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.templatemodel import TemplateModel
from utils import save_hydra_config_to_wandb


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # 1. config logging
    log = logging.getLogger(__name__)
    log.info(f"starting log for: {cfg.group_name}")
    if cfg.log_to_wandb:
        instantiate(cfg.wandb_init)
        save_hydra_config_to_wandb(cfg)

    # 2. get datamodule
    DM: L.LightningDataModule = instantiate(cfg.datamodule_inst)
    log.info("successfully instantiated the datamodule")

    # TEMP: test that dataloader works...
    DM.setup()
    train_dataloader = DM.train_dataloader()
    for x, y in train_dataloader:
        img = x[0, :, :, :]
        mask = y[0, :, :, :]
        print(img.shape, mask.shape)
        _, axarr = plt.subplots(1, 2)
        axarr[1].imshow(np.squeeze(mask.numpy()), cmap="gray")
        axarr[0].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.show()
        plt.close()
        break

    # 3. get model: either instantiate or load saved model
    Model: TemplateModel = instantiate(cfg.model_inst)
    params_str = pprint.pformat(Model.get_params_dict())
    log.info(f"model params:\n{params_str}")

    # 4. train model
    if cfg.stage == "train":
        trainer = L.Trainer()
        trainer.fit(model=Model, datamodule=DM, stage="train")


if __name__ == "__main__":
    main()
