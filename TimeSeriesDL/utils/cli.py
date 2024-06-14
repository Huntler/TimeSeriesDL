"""This module contains an advanced CLI for training a model."""
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from TimeSeriesDL.debug import VisualizeConv

class TSLightningCLI(LightningCLI):
    """The Time-Series CLI adds methods after training such as:
        - visualize first layer
        - visualize test dataset

    Args:
        LightningCLI (LightningCLI): The base class.
    """
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--visualize-layer-0", default=True, action="store_true")
        parser.add_argument("--visualize-test", default=True, action="store_true")

    def after_fit(self):
        """Method executes after fit automatically.
        """
        if self.config["visualize-layer-0"]:
            vis = VisualizeConv(self.model)
            vis.visualize(f"{self.trainer.logger.log_dir}/analysis.png")

        if self.config["visualize-test"]:
            data = self.trainer.predict(self.model, self.trainer.test_dataloaders)
            # TODO: visualize data
