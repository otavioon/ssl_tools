from typing import List, Optional, Tuple, Union
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import os
import contextlib


class MultiCSVHARDataset:
    def __init__(
        self,
        data_path: Union[Path, str],
        features: Union[str, List[str]] = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),
        swapaxes: Tuple[int, int] = (1, 0),
        label: str = None,
        lazy: bool = False,
        cast_to: str = "float32",
        fix_length: bool = False,
    ):
        """A dataset for multiple CSV files with HAR data
        - Each CSV is a single sample with multiple time steps (rows)
        - The columns defines the features (may be transformed to a tensor)
        - Different samples (csvs) may have different number of time steps
        - A 2-element tuple with the data and the label is returned using
            __getitem__ if the label is specified, otherwise only the data.

        Parameters
        ----------
        data : str
            The location of the data
        features: List[str]
            The list of features to use
        swapaxes: Tuple[int, int], optional
            Swap the axes of the data, for each sample, before returning it
        label: str, optional
            Specify the name of the column with the label of the data
        lazy: bool, optional
            If True, the data will be loaded lazily (i.e. the CSV files will be read only when needed)
        cast_to: str, optional
            Cast the numpy data to the specified type
        fix_length: bool, optional
            If True, the data will be padded to the length of the longest
            sample
        """
        self.data_path = Path(data_path)
        self.features = (
            features if isinstance(features, list) else list(features)
        )
        self.swapaxes = swapaxes
        self.label = label
        self.cast_to = cast_to
        self.fix_length = fix_length
        self.files = self._scan_data()
        # Data contains all the data if lazy is False else None
        self._data = self._read_all_csv() if not lazy else None
        self._longest_sample_size = self._get_longest_sample_size()

    @contextlib.contextmanager
    def _disable_fix_length(self):
        """Decorator to disable fix_length when calling a function"""
        old_fix_length = self.fix_length
        self.fix_length = False
        yield
        self.fix_length = old_fix_length

    def _scan_data(self) -> List[Path]:
        """List the CSV files in the data directory

        Returns
        -------
        List[Path]
            List of CSV files
        """
        return list(self.data_path.glob("*.csv"))

    def _get_longest_sample_size(self) -> int:
        """Return the size of the longest sample in the dataset

        Returns
        -------
        int
            The size of the longest sample in the dataset
        """
        if not self.fix_length:
            return 0

        with self._disable_fix_length():
            longest_sample_size = max(
                self[i][0].shape[-1] for i in range(len(self))
            )
        return longest_sample_size

    def _read_csv(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read a single CSV file (a single sample)

        Parameters
        ----------
        path : Path
            The path to the CSV file

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            A 2-element tuple with the data and the label. If the label is not specified, the second element is None.
        """
        # Read the data
        data = pd.read_csv(path)
        # Collect the features
        data = data[self.features].values
        # Cast the data to the specified type
        data = data.astype(self.cast_to)
        # If swap axes is specified, swap the axes
        if self.swapaxes is not None:
            data = np.swapaxes(data, self.swapaxes[0], self.swapaxes[1])

        # Read the label if specified and return the data and the label
        if self.label is not None:
            return data, data[self.label].values
        # If label is not specified, return only the data
        else:
            return data, None

    def _read_all_csv(
        self,
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Read all the CSV files in the data directory

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            A list of 2-element tuple with the data and the label. If the label is not specified, the second element of the tuples are None.
        """
        return [self._read_csv(f) for f in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def _pad_data(self, data: np.ndarray) -> np.ndarray:
        """Pad the data to the length of the longest sample. In summary, this
        function makes the data cyclic.

        Parameters
        ----------
        data : np.ndarray
            The data to pad

        Returns
        -------
        np.ndarray
            The padded data
        """
        time_len = data.shape[-1]

        if time_len == self._longest_sample_size:
            return data

        # Repeat the data along the time axis to match the longest sample size
        repetitions = self._longest_sample_size // time_len + 1
        data = np.tile(data, (1, repetitions))[:, : self._longest_sample_size]
        return data

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Get a single sample from the dataset

        Parameters
        ----------
        idx : int
            The index of the sample

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
            A 2-element tuple with the data and the label if the label is specified, otherwise only the data.
        """
        # If the data is not loaded, load it lazily (read the CSV file)
        if self._data is None:
            data, label = self._read_csv(self.files[idx])
        # Else, read from the loaded data
        else:
            data, label = self._data[idx]

        # Pad the data if fix_length is True
        if self.fix_length:
            data = self._pad_data(data)
            if label is not None:
                label = self._pad_data(label)

        # If label is specified, return the data and the label
        if label is not None:
            return data, label
        # Else, return only the data
        else:
            return data

    def __str__(self) -> str:
        return (
            f"MultiCSVHARDataset at {self.data_path} with {len(self)} samples"
        )

    def __repr__(self) -> str:
        return str(self)


class MultiModalHARDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        batch_size: int = 32,
        num_workers: int = None,
        fix_length: bool = False,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers or os.cpu_count()
        self.fix_length = fix_length

    def _load_dataset(self, name: str):
        path = self.data_path / name
        dataset = MultiCSVHARDataset(
            path, swapaxes=(1, 0), fix_length=self.fix_length
        )
        return dataset

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
