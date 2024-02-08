import logging
import pickle
from pathlib import Path

from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier

from ml_template.datamodules.base import BaseDataMod
from ml_template.models.base import BaseModel
from ml_template.utils import check_dir

log = logging.getLogger(__name__)


class RF(BaseModel):
    def __init__(self, cfg: DictConfig, mode: str, load_model_path: str):
        super().__init__()
        self.mode = mode
        if mode == "trainval":
            self.model = RandomForestClassifier(**cfg.params)
        elif mode == "test":
            self.model = self.loader(load_model_path)

    def trainer(self, DM: BaseDataMod):
        X_train, y_train = DM.X_train, DM.y_train
        self.model.fit(X_train, y_train)

    def predictor(self, X):
        return self.model.predict(X)

    def saver(self, path: str):
        log.info(f"saving model to {path}")
        # make new directory if the parent is missing
        path_parent = Path(path).parent.absolute()
        check_dir(path_parent)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def loader(self, path: str):
        log.info(f"trying to load model from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_params_dict(self):
        return self.model.get_params()
