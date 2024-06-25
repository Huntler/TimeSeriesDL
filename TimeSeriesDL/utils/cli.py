"""This module contains an advanced CLI for training a model."""
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from TimeSeriesDL.debug import VisualizeConv

class TSLightningCLI(LightningCLI):
    """The Time-Series CLI adds methods after training such as:
        - visualize first layer
        - visualize test dataset

        Before training, the user can set options to:
        - find the best lr rate

    Args:
        LightningCLI (LightningCLI): The base class.
    """
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--visualize-layer-0", default=True, action="store_true")
        parser.add_argument("--visualize-test", default=True, action="store_true")
        parser.add_argument("--find-lr", default=False, action="store_true")
        parser.add_argument("--save-every", default=1)

    def before_fit(self):
        """Method executes before fit automatically.
        """
        # find the best lr rate
        if self.config.find_lr:
            tuner = Tuner(self.trainer)
            lr_finder = tuner.lr_find(self.model, self.datamodule, num_training=500, mode="linear",
                                      min_lr=1e-8, max_lr=0.3)
            fig = lr_finder.plot(suggest=True)
            fig.show()

            self.model.lr = lr_finder.suggestion()

        # add model checkpoint callback
        model_check = ModelCheckpoint(self.trainer.logger.log_dir,
                                      every_n_epochs=self.config.save_every)
        for i, c in enumerate(self.trainer.callbacks):
            if isinstance(c, ModelCheckpoint):
                self.trainer.callbacks[i] = model_check

    def after_fit(self):
        """Method executes after fit automatically.
        """
        if self.config.visualize_layer_0:
            vis = VisualizeConv(self.model)
            vis.visualize(f"{self.trainer.logger.log_dir}/analysis.png")

        if self.config.visualize_test:
            pass
            # data = self.trainer.predict(self.model, self.trainer.test_dataloaders, None)
            # TODO: visualize data
