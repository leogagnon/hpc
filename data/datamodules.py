from pytorch_lightning import LightningDataModule
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import *
import torch


class MNISTDataModule(LightningDataModule):
    def __init__(self, train_batch_size: int, val_batch_size: int, data_dir: str = "data/"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.target_transform = Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                0, torch.tensor(y), value=1
            )
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir,
                train=True,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.val_batch_size)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
