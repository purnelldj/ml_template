import logging

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.base import BaseModel
from utils import save_hydra_config_to_wandb

# import pprint


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

    # 3. get model: either instantiate or load saved model
    Model: BaseModel = instantiate(cfg.model_inst)
    # params_str = pprint.pformat(Model.get_params_dict())
    # log.info(f"model params:\n{params_str}")

    # TEMP: test that dataloader works...
    test_dataloader_and_model = False
    if test_dataloader_and_model:
        print("testing dataloader and model..")
        DM.setup(stage="fit")
        train_dataloader = DM.train_dataloader()
        for x, y in train_dataloader:
            img_tr = x[0, :, :, :]
            mask_tr = y[0, :, :, :]
            img = np.transpose(img_tr.numpy(), (1, 2, 0))
            mask = np.squeeze(mask_tr.numpy())
            print(img.shape, mask.shape)
            _, axarr = plt.subplots(1, 2)
            axarr[0].imshow(img)
            axarr[1].imshow(mask, cmap="gray")
            plt.show()
            plt.close()
            # now try run the model forwards
            y_out_tr = Model.forward(x)
            print(y_out_tr.shape)
            img_out = np.transpose(y_out_tr[0, :, :, :].detach().numpy(), (1, 2, 0))
            _, ax = plt.subplots()
            ax.imshow(img_out)
            plt.show()
            plt.close()
            break

    # 4. train model
    if cfg.stage == "train":
        DM.setup(stage="fit")
        log.info("setup datamodule for training")
        trainer = L.Trainer()
        trainer.fit(model=Model, train_dataloaders=DM.train_dataloader())


if __name__ == "__main__":
    main()
