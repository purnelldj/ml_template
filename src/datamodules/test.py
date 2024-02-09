from datamodules.base import BaseDataMod
from omegaconf import DictConfig


class DM(BaseDataMod):

    def __init__(self, cfg: DictConfig, dsub="trainval"):
        print("success")
        print(cfg)
        print(dsub)
        pass

if __name__ == "__main__":
    DM()
