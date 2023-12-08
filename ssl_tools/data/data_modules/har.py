from ssl_tools.data.datasets import (
    MultiModalSeriesCSVDataset,
    SeriesFolderCSVDataset,
    TNCDataset,
    TFCDataset,
)

from torch.utils.data import DataLoader
from typing import Callable, Dict, Union, List

from pathlib import Path
from ssl_tools.transforms.time_1d import AddGaussianNoise
from ssl_tools.transforms.signal_1d import AddRemoveFrequency

import os

import lightning as L


class UserActivityFolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        # Dataset Params
        data_path: Union[Path, str],
        features: List[str] = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),
        label: str = None,
        pad: bool = False,
        transforms: Union[List[Callable], Dict[str, List[Callable]]] = None,
        cast_to: str = "float32",
        # Loader params
        batch_size: int = 1,
        num_workers: int = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.features = features
        self.label = label
        self.pad = pad
        if isinstance(transforms, list) or transforms is None:
            self.transforms = {
                "train": transforms,
                "validation": transforms,
                "test": transforms,
                "predict": transforms,
            }
        elif isinstance(transforms, dict):
            valid_keys = ["train", "validation", "test", "predict"]
            assert all(
                key in valid_keys for key in transforms.keys()
            ), f"Invalid transform key. Must be one of: {valid_keys}"
            self.transforms = transforms

        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.cast_to = cast_to

        self.datasets = {}

    def _load_dataset(self, split_name: str) -> SeriesFolderCSVDataset:
        assert split_name in [
            "train",
            "validation",
            "test",
            "predict",
        ], f"Invalid split_name: {split_name}"

        return SeriesFolderCSVDataset(
            self.data_path / split_name,
            features=self.features,
            label=self.label,
            pad=self.pad,
            transforms=self.transforms[split_name],
            cast_to=self.cast_to,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.datasets["train"] = self._load_dataset("train")
            self.datasets["validation"] = self._load_dataset("validation")
        elif stage == "test":
            self.datasets["test"] = self._load_dataset("test")
        elif stage == "predict":
            self.datasets["predict"] = self._load_dataset("predict")
        else:
            raise ValueError(f"Invalid setup stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class TNCHARDataModule(UserActivityFolderDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        features: List[str] = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),
        label: str = None,
        pad: bool = False,
        transforms: Union[List[Callable], Dict[str, List[Callable]]] = None,
        batch_size: int = 1,
        num_workers: int = None,
        cast_to: str = "float32",
        # TNC parameters
        window_size: int = 60,
        mc_sample_size: int = 20,
        significance_level: float = 0.01,
        repeat: int = 1,
    ):
        super().__init__(
            data_path,
            features=features,
            label=label,
            pad=pad,
            transforms=transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            cast_to=cast_to,
        )

        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.significance_level = significance_level
        self.repeat = repeat

    def _load_dataset(self, split_name: str) -> TNCDataset:
        har_dataset = super()._load_dataset(split_name)
        dataset = TNCDataset(
            har_dataset,
            window_size=self.window_size,
            mc_sample_size=self.mc_sample_size,
            significance_level=self.significance_level,
            repeat=self.repeat,
            cast_to=self.cast_to,
        )
        return dataset


class MultiModalHARSeriesDataModule(L.LightningDataModule):
    def __init__(
        self,
        # Dataset params
        data_path: Union[Path, str],
        feature_prefixes: List[str] = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),
        label: str = "standard activity code",
        features_as_channels: bool = True,
        transforms: Union[List[Callable], Dict[str, List[Callable]]] = None,
        cast_to: str = "float32",
        # Loader params
        batch_size: int = 1,
        num_workers: int = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.feature_prefixes = feature_prefixes
        self.label = label
        self.features_as_channels = features_as_channels
        if isinstance(transforms, list) or transforms is None:
            self.transforms = {
                "train": transforms,
                "validation": transforms,
                "test": transforms,
                "predict": transforms,
            }
        elif isinstance(transforms, dict):
            valid_keys = ["train", "validation", "test", "predict"]
            assert all(
                key in valid_keys for key in transforms.keys()
            ), f"Invalid transform key. Must be one of: {valid_keys}"
            self.transforms = transforms

        self.cast_to = cast_to
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        self.datasets = {}

    def _load_dataset(self, split_name: str) -> SeriesFolderCSVDataset:
        assert split_name in [
            "train",
            "validation",
            "test",
            "predict",
        ], f"Invalid split_name: {split_name}"

        return MultiModalSeriesCSVDataset(
            self.data_path / f"{split_name}.csv",
            feature_prefixes=self.feature_prefixes,
            label=self.label,
            features_as_channels=self.features_as_channels,
            cast_to=self.cast_to,
            transforms=self.transforms[split_name],
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.datasets["train"] = self._load_dataset("train")
            self.datasets["validation"] = self._load_dataset("validation")
        elif stage == "test":
            self.datasets["test"] = self._load_dataset("test")
        elif stage == "predict":
            self.datasets["predict"] = self._load_dataset("predict")
        else:
            raise ValueError(f"Invalid setup stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class TFCDataModule(L.LightningDataModule):
    def __init__(
        self,
        # Dataset Params
        data_path: Union[Path, str],
        feature_prefixes: Union[str, List[str]] = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),
        label: str = "standard activity code",
        features_as_channels: bool = True,
        length_alignment: int = 60,
        time_transforms: Union[
            List[Callable], Dict[str, List[Callable]]
        ] = None,
        frequency_transforms: Union[
            List[Callable], Dict[str, List[Callable]]
        ] = None,
        jitter_ratio: float = 2,
        only_time_frequency: bool = False,
        # Loader params
        batch_size: int = 32,
        num_workers: int = None,
        cast_to: str = "float32",
    ):
        """Define a dataloader for ``TFCDataset``. This is a wrapper around
        ``TFCDataset`` class that defines the dataloaders for Pytorch Lightning.
        The data (``data_path``) must contains three CSV files: train.csv,
        validation.csv and test.csv.

        Parameters
        ----------
        data_path : Union[Path, str]
            The location of the data (root folder). Inside it there must be
            three files: train.csv, validation.csv and test.csv
        batch_size : int, optional
            The size of the batch, by default 1
        num_workers : int, optional
            Number of workers to load data, by default None (use all cores)
        feature_prefixes : Union[str, List[str]], optional
            The prefix of the column names in the dataframe that will be used
            to become features. Used to instantiate ``HARDataset`` dataset.
        label : str, optional
            The label column, by default "standard activity code"
        features_as_channels : bool, optional
            If True, the data will be returned as a vector of shape (C, T),
            where C is the number of features (in feature_prefixes) and T is
            the number of time steps. If False, the data will be returned as a
            vector of shape  T*C. Used to instantiate ``HARDataset`` dataset.
        length_alignment : int, optional
            Truncate the features to this value, by default 178
        time_transforms : Union[Transform, List[Transform]], optional
            List of transforms to apply to the time domain. Used to instantiate
            ``TFCDataset`` dataset. If None. an ``AddGaussianNoise`` transform
            will be used with the given ``jitter_ratio``.
        frequency_transforms : Union[Transform, List[Transform]], optional
            List of transforms to apply to the frequency domain. Used to
            instantiate  ``TFCDataset`` dataset. If None, an
            ``AddRemoveFrequency`` transform will be used.
        cast_to : str, optional
            Cast the data to the given type, by default "float32"
        jitter_ratio : float, optional
            If no time transforms are given (``time_transforms``),
            this parameter will be used to instantiate an ``AddGaussianNoise``
            transform with the given ``jitter_ratio``.
        only_time_frequency : bool, optional
            If True, the data returned will be a 2-element tuple with the
            (time, frequency) data as the first element and the label as the
            second element, by default False
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.feature_prefixes = feature_prefixes
        self.label = label
        self.features_as_channels = features_as_channels
        self.time_transforms = time_transforms
        self.frequency_transforms = frequency_transforms
        self.cast_to = cast_to
        self.length_alignment = length_alignment
        self.only_time_frequency = only_time_frequency

        # Time transforms
        if isinstance(time_transforms, list) or time_transforms is None:
            self.time_transforms = {
                "train": [AddGaussianNoise(std=jitter_ratio)],
                "validation": [AddGaussianNoise(std=jitter_ratio)],
                "test": [AddGaussianNoise(std=jitter_ratio)],
                "predict": [AddGaussianNoise(std=jitter_ratio)],
            }
        elif isinstance(time_transforms, dict):
            self.time_transforms = {
                "train": [AddGaussianNoise(std=jitter_ratio)],
                "validation": [AddGaussianNoise(std=jitter_ratio)],
                "test": [AddGaussianNoise(std=jitter_ratio)],
                "predict": [AddGaussianNoise(std=jitter_ratio)],
            }

            valid_keys = ["train", "validation", "test", "predict"]
            assert all(
                key in valid_keys for key in time_transforms.keys()
            ), f"Invalid transform key. Must be one of: {valid_keys}"
            self.time_transforms.update(time_transforms)

        # Frequency transforms
        if (
            isinstance(frequency_transforms, list)
            or frequency_transforms is None
        ):
            self.frequency_transforms = {
                "train": [AddRemoveFrequency()],
                "validation": [AddRemoveFrequency()],
                "test": [AddRemoveFrequency()],
                "predict": [AddRemoveFrequency()],
            }
        elif isinstance(frequency_transforms, dict):
            self.frequency_transforms = {
                "train": [AddRemoveFrequency()],
                "validation": [AddRemoveFrequency()],
                "test": [AddRemoveFrequency()],
                "predict": [AddRemoveFrequency()],
            }

            valid_keys = ["train", "validation", "test", "predict"]
            assert all(
                key in valid_keys for key in frequency_transforms.keys()
            ), f"Invalid transform key. Must be one of: {valid_keys}"
            self.frequency_transforms.update(frequency_transforms)
            
        self.datasets = {}


    def _load_dataset(self, split_name: str) -> TFCDataset:
        """Load a ``TFCDataset``

        Parameters
        ----------
        name : str
            Name of the split (train, validation or test). This will be used to
            load the corresponding CSV file.

        Returns
        -------
        TFCDataset
            A TFC dataset with the given split.
        """
        assert split_name in [
            "train",
            "validation",
            "test",
            "predict",
        ], f"Invalid split_name: {split_name}"

        
        path = self.data_path / f"{split_name}.csv"
        dataset = MultiModalSeriesCSVDataset(
            data_path=path,
            feature_prefixes=self.feature_prefixes,
            label=self.label,
            features_as_channels=self.features_as_channels,
            cast_to=self.cast_to,
        )
        tfc_dataset = TFCDataset(
            dataset,
            length_alignment=self.length_alignment,
            time_transforms=self.time_transforms[split_name],
            frequency_transforms=self.frequency_transforms[split_name],
            cast_to=self.cast_to,
            only_time_frequency=self.only_time_frequency,
        )
        return tfc_dataset
    
    def setup(self, stage: str):
        if stage == "fit":
            self.datasets["train"] = self._load_dataset("train")
            self.datasets["validation"] = self._load_dataset("validation")
        elif stage == "test":
            self.datasets["test"] = self._load_dataset("test")
        elif stage == "predict":
            self.datasets["predict"] = self._load_dataset("predict")
        else:
            raise ValueError(f"Invalid setup stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )