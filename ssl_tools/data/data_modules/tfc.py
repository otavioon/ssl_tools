from ssl_tools.data.datasets import MultiModalSeriesCSVDataset, TFCDataset

from torch.utils.data import DataLoader
from typing import Union, List

from pathlib import Path

import os

from librep.base import Transform
import lightning as L


from ssl_tools.transforms.time_1d import AddGaussianNoise
from ssl_tools.transforms.signal_1d import AddRemoveFrequency


class TFCDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        batch_size: int = 32,
        num_workers: int = None,
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
        length_alignment: int = 178,
        time_transforms: Union[Transform, List[Transform]] = None,
        frequency_transforms: Union[Transform, List[Transform]] = None,
        cast_to: str = "float32",
        jitter_ratio: float = 2,
        only_time_frequency: bool = False,
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
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.feature_prefixes = feature_prefixes
        self.label = label
        self.features_as_channels = features_as_channels
        self.time_transforms = time_transforms
        self.frequency_transforms = frequency_transforms
        self.cast_to = cast_to
        self.length_alignment = length_alignment
        self.only_time_frequency = only_time_frequency

        if self.time_transforms is None:
            self.time_transforms = [AddGaussianNoise(std=jitter_ratio)]

        if self.frequency_transforms is None:
            self.frequency_transforms = [AddRemoveFrequency()]

    def _load_dataset(self, name: str) -> TFCDataset:
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
        path = self.data_path / f"{name}.csv"
        dataset = MultiModalSeriesCSVDataset(
            path,
            feature_prefixes=self.feature_prefixes,
            label=self.label,
            cast_to=self.cast_to,
            features_as_channels=self.features_as_channels,
        )
        tfc_dataset = TFCDataset(
            dataset,
            length_alignment=self.length_alignment,
            time_transforms=self.time_transforms,
            frequency_transforms=self.frequency_transforms,
            cast_to=self.cast_to,
            only_time_frequency=self.only_time_frequency,
        )
        return tfc_dataset

    def train_dataloader(self) -> DataLoader:
        """Train dataloader for Pytorch Lightning

        Returns
        -------
        DataLoader
            A Dataloader used for training
        """
        dataset = self._load_dataset("train")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader for Pytorch Lightning

        Returns
        -------
        DataLoader
            A Dataloader used for validation
        """
        dataset = self._load_dataset("validation")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader for Pytorch Lightning

        Returns
        -------
        DataLoader
            A Dataloader used for test
        """
        dataset = self._load_dataset("test")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
