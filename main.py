from pytorch_lightning.cli import LightningCLI
from pl_modules import ClassificationIL
from data.datamodules import MNISTDataModule
import torch
import wandb
import jsonargparse
import warnings
import pytorch_lightning as pl


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Global logdir so that everything is in the same place
        parser.add_argument("--logdir", default="log/")

    def before_fit(self):
        # Defaults tensors to GPU if ran on GPU (to avoid a bunch to .to(device))
        if self.config["fit.trainer.accelerator"] == "gpu":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        # Save full config to wandb
        if wandb.run != None:
            wandb.config.update(jsonargparse.namespace_to_dict(self.config["fit"]))


if __name__ == "__main__":

    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    torch.use_deterministic_algorithms(True)

    cli = CLI(
        model_class=ClassificationIL,
        datamodule_class=MNISTDataModule,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
