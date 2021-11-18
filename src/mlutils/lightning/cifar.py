from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mlutils.pytorch.data import DatasetFromSubset, PartiallyLabelledDatasetFromSubset


class CIFARDataModule(pl.LightningDataModule):
    """PyTorch Lightning CIFAR100 data module."""

    def __init__(
        self, batch_size, batch_size_test, dataset_dir, seed=None, version="CIFAR10"
    ):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.dataset_dir = dataset_dir
        self.seed = seed

        assert version in ("CIFAR10", "CIFAR100")
        self.CIFAR = datasets.CIFAR10 if version == "CIFAR10" else datasets.CIFAR100

    def prepare_data(self) -> None:
        """Download the data (if necessary)."""
        self.CIFAR(self.dataset_dir, train=True, download=True)
        self.CIFAR(self.dataset_dir, train=False, download=True)

    def setup(self, stage) -> None:
        """Split and transform."""
        transforms_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transforms_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_set = self.CIFAR(self.dataset_dir, train=True)
        if self.seed:
            train_set, val_set = random_split(
                train_set,
                [45000, 5000],
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            train_set, val_set = random_split(train_set, [45000, 5000])

        self.cifar_train = DatasetFromSubset(train_set, transform=transforms_train)
        self.cifar_val = DatasetFromSubset(val_set, transform=transforms_test)
        self.cifar_test = self.CIFAR(
            self.dataset_dir, train=False, transform=transforms_test
        )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Create train dataloader."""
        cifar_train = DataLoader(
            self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        return cifar_train

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Create valid dataloader."""
        cifar_val = DataLoader(
            self.cifar_val, batch_size=self.batch_size, num_workers=4
        )
        return cifar_val

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Create test dataloader."""
        cifar_test = DataLoader(
            self.cifar_test, batch_size=self.batch_size_test, num_workers=4
        )
        return cifar_test


class PartiallyLabelledCIFARDataModule(CIFARDataModule):
    """Partially labelled PyTorch Lightning CIFAR100 data module."""

    def __init__(
        self,
        batch_size,
        batch_size_test,
        dataset_dir,
        proportion_labelled,
        seed=None,
        version="CIFAR10",
    ):
        super().__init__(batch_size, batch_size_test, dataset_dir, seed, version)
        self.proportion_labelled = proportion_labelled

    def setup(self, stage) -> None:
        """Split and transform."""
        transforms_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transforms_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_set = self.CIFAR(self.dataset_dir, train=True)
        if self.seed:
            train_set, val_set = random_split(
                train_set,
                [45000, 5000],
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            train_set, val_set = random_split(train_set, [45000, 5000])

        self.cifar_train = PartiallyLabelledDatasetFromSubset(
            train_set,
            transform=transforms_train,
            proportion_labelled=self.proportion_labelled,
            seed=self.seed,
        )
        self.cifar_val = DatasetFromSubset(val_set, transform=transforms_test)
        self.cifar_test = self.CIFAR(
            self.dataset_dir, train=False, transform=transforms_test
        )
