import pytorch_lightning as pl
import torch.nn as nn
import wandb
from torchvision.transforms import *
from torchmetrics.classification import MulticlassAccuracy
import torch
from models import SequentialIL
from networks import NETWORKS

class ClassificationIL(pl.LightningModule):
    """
    Trains a nn.Sequential model using inference learning
    """

    def __init__(
        self,
        network: str,
        method: str,
        loss_func: nn.Module,
        init: str,
        weight_clipping: bool,
        gamma: float = None,
        inference_iter: int = None,
    ) -> None:
        super().__init__()

        assert method in [
            "fixed_pred",
            "first_step",
            "exact",
            "iterative",
            "strict",
        ], "<method> not supported"
        assert network in NETWORKS.keys(), "<model> not supported"

        self.model = SequentialIL(
            module_list=NETWORKS[network],
            method=method,
            loss_func=loss_func,
            gamma=gamma,
            n_iter=inference_iter,
            init=init
        )

        self.weight_clipping = weight_clipping
        self.loss_func = loss_func
        self.val_accuracy = MulticlassAccuracy(num_classes=10)

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):

        X, Y = batch

        opt = self.optimizers()
        opt.zero_grad()

        loss, targets = self.model.e_step(X, Y)
        opt.step()
        self.log("train/loss", loss, prog_bar=True)

        # From Kinghorn et al. (2022) [https://arxiv.org/pdf/2208.07114.pdf]
        if self.weight_clipping:
            with torch.no_grad():
                for param in self.model.parameters():
                    param.clamp_(-1,1)
    

    def validation_step(self, batch, batch_idx):
        X, Y = batch

        Y_hat = self.model.forward(X)
        loss = self.loss_func(Y_hat, Y)
        self.val_accuracy.update(torch.argmax(Y_hat, dim=1), torch.argmax(Y, dim=1))

        wandb.log({"val/loss": loss})
        wandb.log({"val/accuracy": self.val_accuracy.compute().item()})
