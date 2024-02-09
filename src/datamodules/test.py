from omegaconf import DictConfig

from datamodules.base import BaseDataMod


class DM(BaseDataMod):

    def __init__(self, cfg: DictConfig, dsub="trainval"):
        print("success")
        print(cfg)
        print(dsub)
        pass


if __name__ == "__main__":
    DM()
