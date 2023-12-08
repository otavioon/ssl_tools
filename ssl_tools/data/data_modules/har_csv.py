from ssl_tools.data.datasets import MultiModalSeriesCSVDataset, TFCDataset

from torch.utils.data import DataLoader
from typing import Union, List

from pathlib import Path

import os

import lightning as L




class HARDataModule(L.LightningDataModule):
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
        cast_to: str = "float32",
    ):
        """Data module for the HAR dataset
        
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
        cast_to : str, optional
            Cast the data to the given type, by default "float32"
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.feature_prefixes = feature_prefixes
        self.label = label
        self.features_as_channels = features_as_channels
        self.cast_to = cast_to

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
        return dataset

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
