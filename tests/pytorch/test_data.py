import numpy as np
import pytest
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Lambda

from mlutils.pytorch.data import PartiallyLabelledDatasetFromSubset


class ToyDataset(Dataset):
    """A dummy dataset class for the tests."""

    def __init__(self):
        self.X = np.ones((10, 3))
        self.y = np.ones(10, dtype=int)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]

    def __len__(self):
        return len(self.X)


@pytest.fixture
def subset():
    dataset = ToyDataset()
    indices = list(range(len(dataset)))
    subset = Subset(dataset, indices)

    return subset


class TestPartiallyLabelledDatasetFromSubset:
    """Tests for the PartiallyLabelledDatasetFromSubset class."""

    def test_init_no_transform_fully_labelled(self, subset):
        new_dataset = PartiallyLabelledDatasetFromSubset(
            subset, proportion_labelled=1.0
        )

        assert len(new_dataset) == len(subset)
        assert all(len(X) == 3 for (X, y) in new_dataset)

        assert all([y == 1 for (X, y) in new_dataset])

    def test_init_no_transform_unlabelled(self, subset):
        new_dataset = PartiallyLabelledDatasetFromSubset(
            subset, proportion_labelled=0.0
        )

        assert len(new_dataset) == len(subset)
        assert all(len(X) == 3 for (X, y) in new_dataset)

        assert sum([y == 1 for (X, y) in new_dataset]) == 0
        assert sum([y == -1 for (X, y) in new_dataset]) == 10
        assert len(new_dataset) == len(subset)

    def test_init_no_transform_partially_labelled(self, subset):
        new_dataset = PartiallyLabelledDatasetFromSubset(
            subset, proportion_labelled=0.5
        )

        assert len(new_dataset) == len(subset)
        assert all(len(X) == 3 for (X, y) in new_dataset)

        assert sum([y == 1 for (X, y) in new_dataset]) == 5
        assert sum([y == -1 for (X, y) in new_dataset]) == 5

    def test_init_transform_labelled(self, subset):
        filter_transform = Lambda(lambda x: x[:1])  # Keep only the first feature
        new_dataset = PartiallyLabelledDatasetFromSubset(
            subset, transform=filter_transform, proportion_labelled=1.0
        )

        assert len(new_dataset) == len(subset)

        assert all([len(X) == 1 for (X, y) in new_dataset])
        assert sum([y == 1 for (X, y) in new_dataset]) == 10

    def test_init_target_transform_partially_labelled(self, subset):
        flip_label_transform = Lambda(
            lambda y: -y + 1
        )  # Flip 0,1 binary classification labels
        new_dataset = PartiallyLabelledDatasetFromSubset(
            subset, target_transform=flip_label_transform, proportion_labelled=0.5
        )

        assert len(new_dataset) == len(subset)
        assert all(len(X) == 3 for (X, y) in new_dataset)

        assert sum([y == -1 for (X, y) in new_dataset]) == 5
        assert sum([y == 0 for (X, y) in new_dataset]) == 5
        assert sum([y == 1 for (X, y) in new_dataset]) == 0
