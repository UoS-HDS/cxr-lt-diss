import os

from dotenv import load_dotenv
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import NeptuneLogger


from src.models.chexfusion import CxrModel, CxrModelFusion
from src.datasets.datamodule import CxrDataModule


load_dotenv()


def get_model_class():
    """returns model class based on STAGE env variable"""
    stage = os.getenv("STAGE", "1")
    if stage == "1":
        return CxrModel
    elif stage == "2":
        return CxrModelFusion
    else:
        raise ValueError(f"Unknown STAGE: {stage}. Expected '1' or '2'.")


class CxrLightningCLI(LightningCLI):
    def before_fit(self):
        if isinstance(self.trainer.logger, NeptuneLogger):
            self.trainer.logger.experiment["train/config"].upload("config.yaml")


def cli_main():
    torch.set_float32_matmul_precision("high")
    print(f"Using {torch.cuda.device_count()} gpus")

    model_class = get_model_class()
    cli = CxrLightningCLI(model_class=model_class, datamodule_class=CxrDataModule)  # noqa


if __name__ == "__main__":
    cli_main()
