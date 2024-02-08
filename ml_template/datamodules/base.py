class BaseDataMod:
    def __init__(self):
        self.X, self.y = None, None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.label_dict = None

    def plot_results(self, y, output_dir_plots: str):
        output_dir_plots = output_dir_plots
        y = y
        raise Exception("this is a placeholder")

    def get_dataloader(self, batch_size):
        self.batch_size = batch_size
        raise Exception("this is a placeholder")
