from lightning.pytorch.cli import LightningCLI


class lci(LightningCLI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.save_config_callback(config_filename="poop.yaml")

    def before_fit(self):
        pass


def main():
    # lci(save_config_callback=None)
    lci(save_config_kwargs={'config_filename': "test.yaml"})


if __name__ == "__main__":
    main()
