# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `seed` argument for CIFAR data modules to ensure reproducibility in train/validation set splits, and in the selection of labelled/unlabelled examples (for partially labelled data modules).
- Store labelled versions of the unlabelled examples in the train set (accessible via `PartiallyLabelledCIFARDataModule.cifar_train_ul_set` or as a DataLoader via `PartiallyLabelledCIFARDataModule.train_ul_set_dataloader()`) for partially labelled data modules (for debugging and evaluation).
- Enable `wandb-dl` to be rerun to download remaining files if the script fails.

### Fixed

- Fix use of output directory for `wandb-dl`.
- Fix parsing of run configs with scalar values for `wandb-dl`.

## [0.0.1] - 2021-09-24

### Added

- Initial release.
- Add custom [Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) classes (`DatasetFromSubset`, `PartiallyLabelledDatasetFromSubset`) for creating fully/partially labelled datasets from [Subsets](https://pytorch.org/docs/stable/data.html?highlight=subset#torch.utils.data.Subset).
- Add partially labelled CIFAR10/100 PyTorch Lightning data module for semi-supervised learning.
- Add functionality and entry point (`wandb-dl`) for exporting [Weights & Biases](https://wandb.ai/site) projects.
