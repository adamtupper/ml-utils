"""Data related code.
"""

import math
import random

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
        self, subset, transform=None, target_transform=None, proportion_labelled=1.0
    ):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform
        self.proportion_labelled = proportion_labelled

        num_labels = math.ceil(self.proportion_labelled * len(self.subset))
        self.labelled_indices = random.sample(range(len(self.subset)), num_labels)

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
