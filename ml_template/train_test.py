import logging
import pprint

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from ml_template.datamodules.base import BaseDataMod
from ml_template.models.base import BaseModel
from ml_template.utils import save_hydra_config_to_wandb


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # 1. config logging
    log = logging.getLogger(__name__)
    log.info(f"starting log for: {cfg.group_name}")
    if cfg.log_to_wandb:
        instantiate(cfg.wandb_init)
        save_hydra_config_to_wandb(cfg)

    # 2. get datamodule
    DM: BaseDataMod = instantiate(cfg.datamodule_inst, dsub=cfg.mode)
    log.info("successfully instantiated the datamodule")

    # 3. get model: either instantiate or load saved model
    Model: BaseModel = instantiate(cfg.model_inst)
    params_str = pprint.pformat(Model.get_params_dict())
    log.info(f"model params:\n{params_str}")

    # 4. train model
    if cfg.mode == "trainval":
        Model.trainer(DM)

    # 5. evaluate model
    metrics = Model.evaluate(cfg.mode, DM, cfg.plot_results, cfg.output_dir_plots)
    if cfg.log_to_wandb:
        wandb.log(metrics)

    # 6. save model
    if cfg.save_model and cfg.mode == "trainval":
        Model.saver(cfg.save_model_path)


if __name__ == "__main__":
    main()
