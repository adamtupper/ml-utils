"""Data related code.
"""
import math

import numpy as np
from torch.utils.data import Dataset


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None, target_transform=None):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y = self.subset[index]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.subset)


class PartiallyLabelledDatasetFromSubset(Dataset):
    def __init__(
        self,
        subset,
        transform=None,
        target_transform=None,
        proportion_labelled=1.0,
        seed=None,
    ):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform
        self.proportion_labelled = proportion_labelled

        num_labels = math.ceil(self.proportion_labelled * len(self.subset))
        if seed:
            rng = np.random.default_rng(seed=seed)
            self.labelled_indices = rng.choice(
                range(len(self.subset)), num_labels, replace=False
            )
        else:
            self.labelled_indices = np.random.choice(
                range(len(self.subset)), num_labels, replace=False
            )

    def __getitem__(self, index):
        x, y = self.subset[index]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        if index not in self.labelled_indices:
            # Remove label
            y = -1

        return x, y

    def __len__(self):
        return len(self.subset)
