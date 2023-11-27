from ssl_tools.data.datasets import HARDataset, TFCDataset

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, List

from pathlib import Path

import numpy as np
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
        time_transforms: Union[Transform, List[Transform]] = None,
        frequency_transforms: Union[Transform, List[Transform]] = None,
        cast_to: str = "float32",
        length_alignment: int = 178,
        jitter_ratio: float = 2
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers or os.cpu_count()
        self.feature_prefixes = feature_prefixes
        self.label = label
        self.features_as_channels = features_as_channels
        self.time_transforms = time_transforms
        self.frequency_transforms = frequency_transforms
        self.cast_to = cast_to
        self.length_alignment = length_alignment

        if self.time_transforms is None:
            self.time_transforms = AddGaussianNoise(std=jitter_ratio)

        if self.frequency_transforms is None:
            self.frequency_transforms = AddRemoveFrequency()


    def _load_dataset(self, name: str):
        path = self.data_path / f"{name}.csv"
        dataset = HARDataset(
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
        )
        return tfc_dataset

    def train_dataloader(self):
        dataset = self._load_dataset("train")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = self._load_dataset("validation")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = self._load_dataset("test")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
