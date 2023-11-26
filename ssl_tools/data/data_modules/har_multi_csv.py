from typing import Union
from pathlib import Path
import os

import lightning as L
from torch.utils.data import DataLoader

from ssl_tools.data.datasets.har_multi_csv import MultiCSVHARDataset
from ssl_tools.data.datasets.tnc_dataset import TNCDataset


class MultiModalHARDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        batch_size: int = 32,
        num_workers: int = None,
        fix_length: bool = False,
        label: str = None,
        cast_to: str = "float32",
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers or os.cpu_count()
        self.fix_length = fix_length
        self.label = label
        self.cast_to = cast_to

    def _load_dataset(self, name: str):
        path = self.data_path / name
        dataset = MultiCSVHARDataset(
            path, swapaxes=(1, 0), fix_length=self.fix_length,
            label=self.label, cast_to=self.cast_to
        )
        return dataset

    def train_dataloader(self):
        dataset = self._load_dataset("train")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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


class TNCHARDataModule(MultiModalHARDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        batch_size: int = 32,
        num_workers: int = None,
        fix_length: bool = False,
        label: str = None,
        cast_to: str = "float32",
        window_size: int = 60,
        mc_sample_size: int = 20,
        significance_level: float = 0.01,
        repeat: int = 1,
    ):
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            fix_length=fix_length,
            label=label,
            cast_to=cast_to,
        )

        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.significance_level = significance_level
        self.repeat = repeat
        self.cast_to = cast_to

    def _load_dataset(self, name: str):
        har_dataset = super()._load_dataset(name)
        dataset = TNCDataset(
            har_dataset,
            window_size=self.window_size,
            mc_sample_size=self.mc_sample_size,
            significance_level=self.significance_level,
            repeat=self.repeat,
            cast_to=self.cast_to,
        )
        return dataset
