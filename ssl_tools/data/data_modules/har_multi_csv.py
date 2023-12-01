from typing import Union
from pathlib import Path
import os

import lightning as L
from torch.utils.data import DataLoader

from ssl_tools.data.datasets.har_multi_csv import MultiCSVHARDataset
from ssl_tools.data.datasets.tnc import TNCDataset


class MultiModalHARDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        batch_size: int = 1,
        num_workers: int = None,
        fix_length: bool = False,
        label: str = None,
        cast_to: str = "float32",
    ):
        """Define train, validation and test dataloaders for HAR datasets in 
        which each sample is a CSV file with multiple time steps (rows) and
        each column is a feature. Note that samples may have different number
        of time steps.
        This is a wrapper around MultiCSVHARDataset that defines the dataloaders
        for Pytorch Lightning. 
        The data must be organized in the following way:
        data_path
        ├── test
        │   ├── sample_1.csv
        │   ├── sample_2.csv
        │   ├── ...
        │   └── sample_n.csv
        ├── train
        │   ├── sample_1.csv
        │   ├── sample_2.csv
        │   ├── ...
        │   └── sample_n.csv
        └── validation
            ├── sample_1.csv
            ├── sample_2.csv
            ├── ...
            └── sample_n.csv

        Parameters
        ----------
        data_path : Union[Path, str]
            The location of the data (root folder). Inside it there must be
            three folders: train, validation and test
        batch_size : int, optional
            The size of the batch, by default 1
        num_workers : int, optional
            Number of workers to load data, by default None (use all cores)
        fix_length : bool, optional
            If True, the samples are padded to the length of the longest sample
            in the dataset. Note that if this parameter is False, batch size 
            must be 1, as samples will have different sizes, by default False.
        label : str, optional
            Specify the column to be used as features. If True, a 2-element
            is returned with the data and the label, otherwise only the data, 
            by default None
        cast_to: str, optional
            Cast the numpy data to the specified type
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.fix_length = fix_length
        self.label = label
        self.cast_to = cast_to
        
        if not self.fix_length:
            assert self.batch_size == 1, (
                "If fix_length is False, batch size must be 1"
            )

    def _load_dataset(self, name: str) -> MultiCSVHARDataset:
        """Load a MultiCSVHARDataset

        Parameters
        ----------
        name : str
            Name of the split (train, validation or test)

        Returns
        -------
        MultiCSVHARDataset
            A MultiCSVHARDataset
        """
        path = self.data_path / name
        dataset = MultiCSVHARDataset(
            path, swapaxes=(1, 0), fix_length=self.fix_length,
            label=self.label, cast_to=self.cast_to
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
        """This class wraps TNCDataset in MultiModalHARDataModule for Pytorch
        Lightning. It is used for pre-training TNC models.
        
        The MuliModalHARDataModule is used to load each sample. TNCDataset 
        encapulates each sample in a TNC sample. The TNC sample is a 3-element
        tuple with the data, the close and distant samples.
        
        Parameters
        ----------
        data_path : Union[Path, str]
            The location of the data (root folder). Inside it there must be
            three folders: train, validation and test
        batch_size : int, optional
            The size of the batch, by default 1
        num_workers : int, optional
            Number of workers to load data, by default None (use all cores)
        fix_length : bool, optional
            If True, the samples are padded to the length of the longest sample
            in the dataset. Note that if this parameter is False, batch size 
            must be 1, as samples will have different sizes, by default False.
        label : str, optional
            Specify the column to be used as features. If True, a 2-element
            is returned with the data and the label, otherwise only the data, 
            by default None
        cast_to: str, optional
            Cast the numpy data to the specified type
        window_size : int
            Size of the window (δ). The window will be centered at t, with 
            window_size / 2 elements before and after t (X[t - δ, t + δ]])
        mc_sample_size : int
            The number of close and distant samples to be selected. This is
            the maximum number of samples that will be selected.
        significance_level: float, optional
            The significance level of the ADF test. It is used to reject the
            null hypothesis of the test if p-value is less than this value, by  
            default 0.01
        repeat : int, optional
            Simple repeat the element of the dataset ``repeat`` times,
            by default 1
        """
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

    def _load_dataset(self, name: str) -> TNCDataset:
        """Load the dataset

        Parameters
        ----------
        name : str
            Name of the split (train, validation or test)

        Returns
        -------
        TNCDataset
            A TNCDataset
        """
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
