from ssl_tools.data.datasets import (
    MultiModalSeriesCSVDataset,
    SeriesFolderCSVDataset,
    TNCDataset,
    TFCDataset,
)

from torch.utils.data import DataLoader
from typing import Callable, Dict, Iterable, Union, List

from pathlib import Path
from ssl_tools.transforms.time_1d import AddGaussianNoise
from ssl_tools.transforms.signal_1d import AddRemoveFrequency

import os
from ssl_tools.utils.types import PathLike

import lightning as L


def parse_transforms(
    transforms: Union[List[Callable], Dict[str, List[Callable]]]
) -> Dict[str, List[Callable]]:
    """Parse the transforms parameter to a dictionary with the split name as
    key and a list of transforms as value.

    Parameters
    ----------
    transforms : Union[List[Callable], Dict[str, List[Callable]]]
        This could be:
        - None: No transforms will be applied
        - List[Callable]: A list of transforms that will be applied to the
            data. The same transforms will be applied to all splits.
        - Dict[str, List[Callable]]: A dictionary with the split name as
            key and a list of transforms as value. The split name must be
            one of: "train", "validation", "test" or "predict".

    Returns
    -------
    Dict[str, List[Callable]]
        A dictionary with the split name as key and a list of transforms as
        value.
    """
    if isinstance(transforms, list) or transforms is None:
        return {
            "train": transforms,
            "validation": transforms,
            "test": transforms,
            "predict": transforms,
        }
    elif isinstance(transforms, dict):
        # Check if the keys are valid
        valid_keys = ["train", "validation", "test", "predict"]
        assert all(
            key in valid_keys for key in transforms.keys()
        ), f"Invalid transform key. Must be one of: {valid_keys}"
        new_transforms = {
            "train": None,
            "validation": None,
            "test": None,
            "predict": None,
        }
        new_transforms.update(transforms)
        return new_transforms


def parse_num_workers(num_workers: int) -> int:
    """Parse the num_workers parameter. If None, use all cores.

    Parameters
    ----------
    num_workers : int
        Number of workers to load data. If None, then use all cores

    Returns
    -------
    int
        Number of workers to load data.
    """
    return num_workers if num_workers is not None else os.cpu_count()


class UserActivityFolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        # Dataset Params
        data_path: PathLike,
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
        """Define the dataloaders for train, validation and test splits for
        HAR datasets. The data must be in the following folder structure:
        It is a wrapper around ``SeriesFolderCSVDataset`` dataset class.
        The ``SeriesFolderCSVDataset`` class assumes that the data is in a
        folder with multiple CSV files. Each CSV file is a single sample that
        can be composed of multiple time steps (rows). Each column is a feature
        of the sample.

        For instance, if we have two samples, user-1.csv and user-2.csv,
        the directory structure will look something like:

        data_path
        ├── user-1.csv
        └── user-2.csv

        And the data will look something like:
        - user-1.csv:
            +---------+---------+--------+
            | accel-x | accel-y | class  |
            +---------+---------+--------+
            | 0.502123| 0.02123 | 1      |
            | 0.682012| 0.02123 | 1      |
            | 0.498217| 0.00001 | 1      |
            +---------+---------+--------+
        - user-2.csv:
            +---------+---------+--------+
            | accel-x | accel-y | class  |
            +---------+---------+--------+
            | 0.502123| 0.02123 | 0      |
            | 0.682012| 0.02123 | 0      |
            | 0.498217| 0.00001 | 0      |
            | 3.141592| 1.414141| 0      |
            +---------+---------+--------+

        The ``features`` parameter is used to select the columns that will be
        used as features. For instance, if we want to use only the accel-x
        column, we can set ``features=["accel-x"]``. If we want to use both
        accel-x and accel-y, we can set ``features=["accel-x", "accel-y"]``.

        The label column is specified by the ``label`` parameter. Note that we
        have one label per time-step and not a single label per sample.

        The dataset will return a 2-element tuple with the data and the label,
        if the ``label`` parameter is specified, otherwise return only the data.


        Parameters
        ----------
        data_path : PathLike
            The location of the directory with CSV files.
        features: List[str]
            A list with column names that will be used as features. If None,
            all columns except the label will be used as features.
        pad: bool, optional
            If True, the data will be padded to the length of the longest
            sample. Note that padding will be applyied after the transforms,
            and also to the labels if specified.
        label: str, optional
            Specify the name of the column with the label of the data
        transforms : Union[List[Callable], Dict[str, List[Callable]]], optional
            This could be:
            - None: No transforms will be applied
            - List[Callable]: A list of transforms that will be applied to the
                data. The same transforms will be applied to all splits.
            - Dict[str, List[Callable]]: A dictionary with the split name as
                key and a list of transforms as value. The split name must be
                one of: "train", "validation", "test" or "predict".
        cast_to: str, optional
            Cast the numpy data to the specified type
        batch_size : int, optional
            The size of the batch
        num_workers : int, optional
            Number of workers to load data. If None, then use all cores
        """
        super().__init__()

        # ---- Dataset Parameters ----
        # Allowing multiple datasets
        self.data_path = Path(data_path)
        self.features = features
        self.label = label
        self.pad = pad
        self.transforms = parse_transforms(transforms)

        # ---- Loader Parameters ----
        self.batch_size = batch_size
        self.num_workers = parse_num_workers(num_workers)
        self.cast_to = cast_to

        # ---- Class specific ----
        self.datasets = {}

    def _load_dataset(self, split_name: str) -> SeriesFolderCSVDataset:
        """Create a ``SeriesFolderCSVDataset`` dataset with the given split.

        Parameters
        ----------
        split_name : str
            Name of the split (train, validation or test). This will be used to
            load the corresponding CSV file.

        Returns
        -------
        SeriesFolderCSVDataset
            The dataset with the given split.
        """
        assert split_name in [
            "train",
            "validation",
            "test",
            "predict",
        ], f"Invalid split_name: {split_name}"
        
        if split_name == "predict":
            split_name = "test"

        return SeriesFolderCSVDataset(
            self.data_path / split_name,
            features=self.features,
            label=self.label,
            pad=self.pad,
            transforms=self.transforms[split_name],
            cast_to=self.cast_to,
        )

    def setup(self, stage: str):
        """Assign the datasets to the corresponding split. ``self.datasets``
        will be a dictionary with the split name as key and the dataset as
        value.

        Parameters
        ----------
        stage : str
            The stage of the setup. This could be:
            - "fit": Load the train and validation datasets
            - "test": Load the test dataset
            - "predict": Load the predict dataset

        Raises
        ------
        ValueError
            If the stage is not one of: "fit", "test" or "predict"
        """
        if stage == "fit":
            self.datasets["train"] = self._load_dataset("train")
            self.datasets["validation"] = self._load_dataset("validation")
        elif stage == "test":
            self.datasets["test"] = self._load_dataset("test")
        elif stage == "predict":
            self.datasets["predict"] = self._load_dataset("test")
        else:
            raise ValueError(f"Invalid setup stage: {stage}")

    def _get_loader(
        self, split_name: str, shuffle: bool
    ) -> DataLoader:
        """Get a dataloader for the given split.

        Parameters
        ----------
        split_name : str
            The name of the split. This must be one of: "train", "validation",
            "test" or "predict".
        shuffle : bool
            Shuffle the data or not.

        Returns
        -------
        DataLoader
            A dataloader for the given split.
        """
        return DataLoader(
            self.datasets[split_name],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_loader("validation", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_loader("predict", shuffle=False)
    
    def __str__(self):
        return f"UserActivityFolderDataModule(data_path={self.data_path}, batch_size={self.batch_size})"
    
    def __repr__(self) -> str:
        return str(self)


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
        """Define the dataloaders for train, validation and test splits for
        TNC datasets. The data must be in the following folder structure:
        It is a wrapper around ``TNCDataset`` dataset class.
        The ``SeriesFolderCSVDataset`` class assumes that the data is in a
        folder with multiple CSV files. Each CSV file is a single sample that
        can be composed of multiple time steps (rows). Each column is a feature
        of the sample.

        For instance, if we have two samples, user-1.csv and user-2.csv,
        the directory structure will look something like:

        data_path
        ├── user-1.csv
        └── user-2.csv

        And the data will look something like:
        - user-1.csv:
            +---------+---------+--------+
            | accel-x | accel-y | class  |
            +---------+---------+--------+
            | 0.502123| 0.02123 | 1      |
            | 0.682012| 0.02123 | 1      |
            | 0.498217| 0.00001 | 1      |
            +---------+---------+--------+
        - user-2.csv:
            +---------+---------+--------+
            | accel-x | accel-y | class  |
            +---------+---------+--------+
            | 0.502123| 0.02123 | 0      |
            | 0.682012| 0.02123 | 0      |
            | 0.498217| 0.00001 | 0      |
            | 3.141592| 1.414141| 0      |
            +---------+---------+--------+

        The ``features`` parameter is used to select the columns that will be
        used as features. For instance, if we want to use only the accel-x
        column, we can set ``features=["accel-x"]``. If we want to use both
        accel-x and accel-y, we can set ``features=["accel-x", "accel-y"]``.

        The label column is specified by the ``label`` parameter. Note that we
        have one label per time-step and not a single label per sample.

        The dataset will return a 2-element tuple with the data and the label,
        if the ``label`` parameter is specified, otherwise return only the data.


        Parameters
        ----------
        data_path : PathLike
            The location of the directory with CSV files.
        features: List[str]
            A list with column names that will be used as features. If None,
            all columns except the label will be used as features.
        pad: bool, optional
            If True, the data will be padded to the length of the longest
            sample. Note that padding will be applyied after the transforms,
            and also to the labels if specified.
        label: str, optional
            Specify the name of the column with the label of the data
        transforms : Union[List[Callable], Dict[str, List[Callable]]], optional
            This could be:
            - None: No transforms will be applied
            - List[Callable]: A list of transforms that will be applied to the
                data. The same transforms will be applied to all splits.
            - Dict[str, List[Callable]]: A dictionary with the split name as
                key and a list of transforms as value. The split name must be
                one of: "train", "validation", "test" or "predict".
        cast_to: str, optional
            Cast the numpy data to the specified type
        batch_size : int, optional
            The size of the batch
        num_workers : int, optional
            Number of workers to load data. If None, then use all cores
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
        """
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
        """Create a ``TNCDataset`` dataset with the given split.

        Parameters
        ----------
        split_name : str
            The name of the split. This must be one of: "train", "validation",
            "test" or "predict".

        Returns
        -------
        TNCDataset
            A TNC dataset with the given split.
        """
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
        data_path: PathLike,
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
        """Define the dataloaders for train, validation and test splits for
        HAR datasets. This datasets assumes that the data is in a single CSV
        file with series of data. Each row is a single sample that can be
        composed of multiple modalities (series). Each column is a feature of
        some series with the prefix indicating the series. The suffix may
        indicates the time step. For instance, if we have two series, accel-x
        and accel-y, the data will look something like:

        +-----------+-----------+-----------+-----------+--------+
        | accel-x-0 | accel-x-1 | accel-y-0 | accel-y-1 |  class |
        +-----------+-----------+-----------+-----------+--------+
        | 0.502123  | 0.02123   | 0.502123  | 0.502123  |  0     |
        | 0.6820123 | 0.02123   | 0.502123  | 0.502123  |  1     |
        | 0.498217  | 0.00001   | 1.414141  | 3.141592  |  2     |
        +-----------+-----------+-----------+-----------+--------+

        The ``feature_prefixes`` parameter is used to select the columns that
        will be used as features. For instance, if we want to use only the
        accel-x series, we can set ``feature_prefixes=["accel-x"]``. If we want
        to use both accel-x and accel-y, we can set
        ``feature_prefixes=["accel-x", "accel-y"]``. If None is passed, all
        columns will be used as features, except the label column.
        The label column is specified by the ``label`` parameter.

        The dataset will return a 2-element tuple with the data and the label,
        if the ``label`` parameter is specified, otherwise return only the data.

        If ``features_as_channels`` is ``True``, the data will be returned as a
        vector of shape `(C, T)`, where C is the number of channels (features)
        and `T` is the number of time steps. Else, the data will be returned as
        a vector of shape  T*C (a single vector with all the features).

        Parameters
        ----------
        data_path : PathLike
            The path to the folder with "train.csv", "validation.csv" and
            "test.csv" files inside it.
        feature_prefixes : Union[str, List[str]], optional
            The prefix of the column names in the dataframe that will be used
            to become features. If None, all columns except the label will be
            used as features.
        label : str, optional
            The name of the column that will be used as label
        features_as_channels : bool, optional
            If True, the data will be returned as a vector of shape (C, T),
            else the data will be returned as a vector of shape  T*C.
        cast_to: str, optional
            Cast the numpy data to the specified type
        transforms : Union[List[Callable], Dict[str, List[Callable]]], optional
            This could be:
            - None: No transforms will be applied
            - List[Callable]: A list of transforms that will be applied to the
                data. The same transforms will be applied to all splits.
            - Dict[str, List[Callable]]: A dictionary with the split name as
                key and a list of transforms as value. The split name must be
                one of: "train", "validation", "test" or "predict".
        batch_size : int, optional
            The size of the batch
        num_workers : int, optional
            Number of workers to load data. If None, then use all cores
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.feature_prefixes = feature_prefixes
        self.label = label
        self.features_as_channels = features_as_channels
        self.transforms = parse_transforms(transforms)
        self.cast_to = cast_to
        self.batch_size = batch_size
        self.num_workers = parse_num_workers(num_workers)

        self.datasets = {}

    def _load_dataset(self, split_name: str) -> MultiModalSeriesCSVDataset:
        """Create a ``MultiModalSeriesCSVDataset`` dataset with the given split.

        Parameters
        ----------
        split_name : str
            The name of the split. This must be one of: "train", "validation",
            "test" or "predict".

        Returns
        -------
        MultiModalSeriesCSVDataset
            A MultiModalSeriesCSVDataset dataset with the given split.
        """
        assert split_name in [
            "train",
            "validation",
            "test",
            "predict",
        ], f"Invalid split_name: {split_name}"
        
        if split_name == "predict":
            split_name = "test"

        return MultiModalSeriesCSVDataset(
            self.data_path / f"{split_name}.csv",
            feature_prefixes=self.feature_prefixes,
            label=self.label,
            features_as_channels=self.features_as_channels,
            cast_to=self.cast_to,
            transforms=self.transforms[split_name],
        )

    def setup(self, stage: str):
        """Assign the datasets to the corresponding split. ``self.datasets``
        will be a dictionary with the split name as key and the dataset as
        value.

        Parameters
        ----------
        stage : str
            The stage of the setup. This could be:
            - "fit": Load the train and validation datasets
            - "test": Load the test dataset
            - "predict": Load the predict dataset

        Raises
        ------
        ValueError
            If the stage is not one of: "fit", "test" or "predict"
        """
        if stage == "fit":
            self.datasets["train"] = self._load_dataset("train")
            self.datasets["validation"] = self._load_dataset("validation")
        elif stage == "test":
            self.datasets["test"] = self._load_dataset("test")
        elif stage == "predict":
            self.datasets["predict"] = self._load_dataset("predict")
        else:
            raise ValueError(f"Invalid setup stage: {stage}")

    def _get_loader(
        self, split_name: str, shuffle: bool
    ) -> DataLoader:
        """Get a dataloader for the given split.

        Parameters
        ----------
        split_name : str
            The name of the split. This must be one of: "train", "validation",
            "test" or "predict".
        shuffle : bool
            Shuffle the data or not.

        Returns
        -------
        DataLoader
            A dataloader for the given split.
        """
        return DataLoader(
            self.datasets[split_name],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_loader("validation", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_loader("predict", shuffle=False)
    
    def __str__(self):
        return f"MultiModalHARSeriesDataModule(data_path={self.data_path}, batch_size={self.batch_size})"
    
    def __repr__(self) -> str:
        return str(self)


class TFCDataModule(L.LightningDataModule):
    def __init__(
        self,
        # Dataset Params
        data_path: PathLike,
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
        cast_to: str = "float32",
        jitter_ratio: float = 2,
        only_time_frequency: bool = False,
        # Loader params
        batch_size: int = 32,
        num_workers: int = None,
    ):
        """Define a dataloader for ``TFCDataset``. This is a wrapper around
        ``TFCDataset`` class that defines the dataloaders for Pytorch Lightning.
        The data (``data_path``) must contains three CSV files: train.csv,
        validation.csv and test.csv.

        Parameters
        ----------
        data_path : PathLike
            The location of the data (root folder). Inside it there must be
            three files: train.csv, validation.csv and test.csv
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
        time_transforms : Union[List[Callable], Dict[str, List[Callable]]], optional
            Transforms to be applied to time domain data. This could be:
            - None: No transforms will be applied
            - List[Callable]: A list of transforms that will be applied to the
                data. The same transforms will be applied to all splits.
            - Dict[str, List[Callable]]: A dictionary with the split name as
                key and a list of transforms as value. The split name must be
                one of: "train", "validation", "test" or "predict".
            If None. an ``AddGaussianNoise`` transform will be used with the 
            given ``jitter_ratio``.
        frequency_transforms : Union[List[Callable], Dict[str, List[Callable]]], optional
            Transforms to be applied to frequency domain data. This could be:
            - None: No transforms will be applied
            - List[Callable]: A list of transforms that will be applied to the
                data. The same transforms will be applied to all splits.
            - Dict[str, List[Callable]]: A dictionary with the split name as
                key and a list of transforms as value. The split name must be
                one of: "train", "validation", "test" or "predict".            
            If None, an ``AddRemoveFrequency`` transform will be used.
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
        batch_size : int, optional
            The size of the batch, by default 1
        num_workers : int, optional
            Number of workers to load data, by default None (use all cores)
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
        """Create a ``TFCDataset``

        Parameters
        ----------
        split_name : str
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
        
        if split_name == "predict":
            split_name = "test"

        path = self.data_path / f"{split_name}.csv"
        
        # Creates a MultiModalSeriesCSVDataset
        dataset = MultiModalSeriesCSVDataset(
            data_path=path,
            feature_prefixes=self.feature_prefixes,
            label=self.label,
            features_as_channels=self.features_as_channels,
            cast_to=self.cast_to,
        )
        
        # Wraps the MultiModalSeriesCSVDataset with a TFCDataset
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
        """Assign the datasets to the corresponding split. ``self.datasets``
        will be a dictionary with the split name as key and the dataset as
        value.

        Parameters
        ----------
        stage : str
            The stage of the setup. This could be:
            - "fit": Load the train and validation datasets
            - "test": Load the test dataset
            - "predict": Load the predict dataset

        Raises
        ------
        ValueError
            If the stage is not one of: "fit", "test" or "predict"
        """
        if stage == "fit":
            self.datasets["train"] = self._load_dataset("train")
            self.datasets["validation"] = self._load_dataset("validation")
        elif stage == "test":
            self.datasets["test"] = self._load_dataset("test")
        elif stage == "predict":
            self.datasets["predict"] = self._load_dataset("predict")
        else:
            raise ValueError(f"Invalid setup stage: {stage}")

    def _get_loader(
        self, split_name: str, shuffle: bool
    ) -> DataLoader:
        """Get a dataloader for the given split.

        Parameters
        ----------
        split_name : str
            The name of the split. This must be one of: "train", "validation",
            "test" or "predict".
        shuffle : bool
            Shuffle the data or not.

        Returns
        -------
        DataLoader
            A dataloader for the given split.
        """
        return DataLoader(
            self.datasets[split_name],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_loader("validation", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_loader("predict", shuffle=False)