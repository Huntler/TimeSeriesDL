"""Train a model using a CLI or provide the hyperparameters directly to the CLI."""
from lightning.pytorch.loggers import TensorBoardLogger

# simple demo classes for your convenience
from TimeSeriesDL.model import BaseModel
from TimeSeriesDL.data import TSDataModule
from TimeSeriesDL.utils import TSLightningCLI


def cli_main():
    cli = TSLightningCLI(BaseModel, TSDataModule, subclass_mode_model=True, run=False)

    # set up the logger
    logger = TensorBoardLogger("lightning_logs", name=cli.model.__class__.__name__)
    cli.trainer.logger = logger

    # run methods before fit
    cli.before_fit()

    # train
    cli.trainer.fit(cli.model, cli.datamodule)

    # run methods after fit
    cli.after_fit()


if __name__ == "__main__":
    cli_main()
