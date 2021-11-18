import shutil

import pytest
import torch
from pytorch_lightning.utilities.seed import seed_everything
from torchvision import datasets

from mlutils.lightning.cifar import PartiallyLabelledCIFARDataModule

TEMP_DIR_PATH = "tests/tmpdir"


@pytest.fixture
def tmpdir():
    return TEMP_DIR_PATH


class TestPartiallyLabelledCIFARDataModule:
    """Tests for the PartiallyLabelledCIFARDataModule class."""

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(TEMP_DIR_PATH)

    def setup_method(self, method):
        seed_everything(56)

    def test_prepare_data_cifar10(self, tmpdir):
        cifar10_dm = PartiallyLabelledCIFARDataModule(
            batch_size=16,
            batch_size_test=16,
            dataset_dir=tmpdir,
            proportion_labelled=1.0,
        )
        cifar10_dm.prepare_data()

        assert cifar10_dm.CIFAR == datasets.CIFAR10

    def test_prepare_data_cifar100(self, tmpdir):
        cifar100_dm = PartiallyLabelledCIFARDataModule(
            batch_size=16,
            batch_size_test=16,
            dataset_dir=tmpdir,
            proportion_labelled=1.0,
            version="CIFAR100",
        )
        cifar100_dm.prepare_data()

        assert cifar100_dm.CIFAR == datasets.CIFAR100

    def test_setup_cifar10_fully_labelled(self, tmpdir):
        cifar10_dm = PartiallyLabelledCIFARDataModule(
            batch_size=16,
            batch_size_test=16,
            dataset_dir=tmpdir,
            seed=42,
            proportion_labelled=1.0,
        )
        cifar10_dm.prepare_data()
        cifar10_dm.setup(stage=None)

        # Check dataset splits are the correct sizes
        assert len(cifar10_dm.cifar_train) == 45000
        assert len(cifar10_dm.cifar_val) == 5000
        assert len(cifar10_dm.cifar_test) == 10000

        # Check the transforms have been applied
        assert all([type(X) == torch.Tensor for (X, y) in cifar10_dm.cifar_train])
        assert all([type(X) == torch.Tensor for (X, y) in cifar10_dm.cifar_val])
        assert all([type(X) == torch.Tensor for (X, y) in cifar10_dm.cifar_test])

        # Check the labels are unchanged
        assert all([0 <= y <= 9 for (X, y) in cifar10_dm.cifar_train])
        assert all([0 <= y <= 9 for (X, y) in cifar10_dm.cifar_val])
        assert all([0 <= y <= 9 for (X, y) in cifar10_dm.cifar_test])

    def test_setup_cifar10_partially_labelled(self, tmpdir):
        cifar10_dm = PartiallyLabelledCIFARDataModule(
            batch_size=16,
            batch_size_test=16,
            dataset_dir=tmpdir,
            seed=42,
            proportion_labelled=0.5,
        )
        cifar10_dm.prepare_data()
        cifar10_dm.setup(stage=None)

        # Check dataset splits are the correct sizes
        assert len(cifar10_dm.cifar_train) == 45000
        assert len(cifar10_dm.cifar_val) == 5000
        assert len(cifar10_dm.cifar_test) == 10000

        # Check the transforms have been applied
        assert all([type(X) == torch.Tensor for (X, y) in cifar10_dm.cifar_train])
        assert all([type(X) == torch.Tensor for (X, y) in cifar10_dm.cifar_val])
        assert all([type(X) == torch.Tensor for (X, y) in cifar10_dm.cifar_test])

        # Check only 50% of the labels in the training set have been modified
        assert all([-1 <= y <= 9 for (X, y) in cifar10_dm.cifar_train])
        assert all([0 <= y <= 9 for (X, y) in cifar10_dm.cifar_val])
        assert all([0 <= y <= 9 for (X, y) in cifar10_dm.cifar_test])
        assert sum([y == -1 for (X, y) in cifar10_dm.cifar_train]) / len(
            cifar10_dm.cifar_train
        ) == pytest.approx(0.5, abs=0.01)

    def test_train_dataloader_cifar100(self, tmpdir):
        cifar100_dm = PartiallyLabelledCIFARDataModule(
            batch_size=1,
            batch_size_test=1,
            dataset_dir=tmpdir,
            seed=42,
            version="CIFAR100",
            proportion_labelled=1.0,
        )
        cifar100_dm.prepare_data()
        cifar100_dm.setup(stage=None)

        assert len(cifar100_dm.train_dataloader()) == len(cifar100_dm.cifar_train)

    def test_val_dataloader_cifar100(self, tmpdir):
        cifar100_dm = PartiallyLabelledCIFARDataModule(
            batch_size=1,
            batch_size_test=1,
            dataset_dir=tmpdir,
            seed=42,
            version="CIFAR100",
            proportion_labelled=1.0,
        )
        cifar100_dm.prepare_data()
        cifar100_dm.setup(stage=None)

        assert len(cifar100_dm.val_dataloader()) == len(cifar100_dm.cifar_val)

    def test_test_dataloader_cifar100(self, tmpdir):
        cifar100_dm = PartiallyLabelledCIFARDataModule(
            batch_size=1,
            batch_size_test=1,
            dataset_dir=tmpdir,
            seed=42,
            version="CIFAR100",
            proportion_labelled=1.0,
        )
        cifar100_dm.prepare_data()
        cifar100_dm.setup(stage=None)

        assert len(cifar100_dm.test_dataloader()) == len(cifar100_dm.cifar_test)

    def test_deterministic_dataset_splitting(self, tmpdir):
        # Prepare first data module
        dm1 = PartiallyLabelledCIFARDataModule(
            batch_size=1,
            batch_size_test=1,
            dataset_dir=tmpdir,
            seed=42,
            version="CIFAR10",
            proportion_labelled=1.0,
        )
        dm1.prepare_data()
        dm1.setup(stage=None)

        # Prepare a second data module
        dm2 = PartiallyLabelledCIFARDataModule(
            batch_size=1,
            batch_size_test=1,
            dataset_dir=tmpdir,
            seed=42,
            version="CIFAR10",
            proportion_labelled=1.0,
        )
        dm2.prepare_data()
        dm2.setup(stage=None)

        # Check the validation sets are equal
        assert dm1.cifar_val.subset.indices == dm2.cifar_val.subset.indices
