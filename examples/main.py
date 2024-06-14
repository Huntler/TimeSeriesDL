# main.py
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

# simple demo classes for your convenience
from TimeSeriesDL.model import ConvLSTM
from TimeSeriesDL.data import TSDataModule
from TimeSeriesDL.debug import VisualizeConv


def cli_main():
    cli = LightningCLI(ConvLSTM, TSDataModule, run=False)

    # set up the logger
    logger = TensorBoardLogger("lightning_logs", name=cli.model.__class__.__name__)
    cli.trainer.logger = logger

    # train
    cli.trainer.fit(cli.model, cli.datamodule)

    # visualize the model
    vis = VisualizeConv(cli.model)
    vis.visualize(f"{logger.log_dir}/analysis.png")


if __name__ == "__main__":
    cli_main()
