import logging
import pprint

from sklearn.metrics import classification_report

from datamodules.base import BaseDataMod
from utils import check_dir


class BaseModel:
    def __init__(self):
        self.model_name = None
        self.model = None

    def trainer(self, DM: BaseDataMod):
        raise Exception("this is a placeholder")

    def predictor(self, X: list):
        raise Exception("this is a placeholder")

    def saver(self, path: str):
        raise Exception("this is a placeholder")

    def loader(self, path: str):
        raise Exception("this is a placeholder")

    def get_params_dict(self):
        raise Exception("this is a placeholder")

    def evaluate(
        self, mode: str, DM: BaseDataMod, plot_results: bool, output_dir_plots: str
    ):
        log = logging.getLogger(__name__)

        metrics = {}
        log.info(f"logging metrics for mode : {mode}")

        if mode == "trainval":
            X_train, y_train = DM.X_train, DM.y_train
            X_val, y_val = DM.X_val, DM.y_val
            y_train_pred = self.predictor(X_train)
            y_val_pred = self.predictor(X_val)
            metrics["train"] = get_metrics(y_train, y_train_pred, DM.label_dict)
            metrics["val"] = get_metrics(y_val, y_val_pred, DM.label_dict)
            if plot_results:
                X, y = DM.X, DM.y
                y_pred = self.predictor(X)
                output_dir_plots_true = output_dir_plots + "true/"
                output_dir_plots_pred = output_dir_plots + "pred/"
                check_dir(output_dir_plots_true)
                check_dir(output_dir_plots_pred)
                DM.plot_results(y, output_dir_plots + "true/")
                DM.plot_results(y_pred, output_dir_plots + "pred/")
        elif mode == "test":
            X_test, y_test = DM.X_test, DM.y_test
            y_test_pred = self.predictor(X_test)
            metrics["test"] = get_metrics(y_test, y_test_pred, DM.label_dict)
            if plot_results:
                output_dir_plots_true = output_dir_plots + "true/"
                output_dir_plots_pred = output_dir_plots + "pred/"
                check_dir(output_dir_plots_true)
                check_dir(output_dir_plots_pred)
                DM.plot_results(y_test, output_dir_plots + "true/")
                DM.plot_results(y_test_pred, output_dir_plots + "pred/")

        metrics_str = pprint.pformat(metrics)
        log.info(f"\n{metrics_str}")

        return metrics


def get_metrics(y_true, y_pred, label_dict: dict):
    """
    y_true and y_pred are array-like
    """
    # better to use the classification report than individual metrics
    # metrics["accuracy"] = accuracy_score(y_true, y_pred)
    # metrics["f1"] = f1_score(y_true, y_pred, pos_label=pos_label)
    # metrics["precision"] = precision_score(y_true, y_pred, pos_label=pos_label)
    # metrics["recall"] = recall_score(y_true, y_pred, pos_label=pos_label)
    target_names = [lbl for lbl in label_dict]
    label_vals = [label_dict[lbl] for lbl in label_dict]
    metrics = classification_report(
        y_true, y_pred, labels=label_vals, target_names=target_names, output_dict=True
    )
    counter = {}
    for name, val in zip(target_names, label_vals):
        counter[f"{name}_true"] = len(y_true[y_true == val])
        counter[f"{name}_pred"] = len(y_pred[y_pred == val])
    metrics["counter"] = counter

    return metrics
