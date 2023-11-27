from typing import List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import contextlib


class HARDataset:
    def __init__(
        self,
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
        cast_to: str = "float32",
        features_as_channels: bool = True,
    ):
        """A dataset for HAR data in CSV format. The data is a single CSV file
        with windows of data. Each row is has a window and each column is a
        feature with a suffix indicating the time step. Something like:
        +-----------+-----------+-----------+-----------+-----------+--------+
        | accel-x-0 | accel-x-1 | accel-x-2 | accel-y-0 | accel-y-1 |  ...   |
        +-----------+-----------+-----------+-----------+-----------+--------+
        | 0.502123  | 0.02123   | 0.502123  | 0.502123  | 0.502123  |  ...   |
        | 0.6820123 | 0.02123   | 0.502123  | 0.502123  | 0.502123  |  ...   |
        +-----------+-----------+-----------+-----------+-----------+--------+

        The dataset will return a 2-element tuple with the data and the label,
        if the ``label`` parameter is specified, otherwise return only the data.

        If ``features_as_channels`` is ``True``, the data will be returned as a
        vector of shape `(C, T)`, where C is the number of channels (features)
        and `T` is the number of time steps. Else, the data will be returned as
        a vector of shape  `T*C`.


        Parameters
        ----------
        data_path : Union[Path, str]
            The location of the CSV file
        feature_prefixes : Union[str, List[str]], optional
            The prefix of the feature columns that will be used
        label : str, optional
            The label column, by default "standard activity code"
        cast_to: str, optional
            Cast the numpy data to the specified type
        features_as_channels : bool, optional
            If True, the data will be returned as a vector of shape (C, T),
            where C is the number of features (in feature_prefixes) and T is
            the number of time steps. If False, the data will be returned as a
            vector of shape  T*C.
        """
        self.data_path = Path(data_path)
        self.feature_prefixes = (
            feature_prefixes
            if isinstance(feature_prefixes, list)
            else list(feature_prefixes)
        )
        self.label = label
        self.cast_to = cast_to
        self.features_as_channels = features_as_channels
        self.data, self.labels = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load data from the CSV file

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            A 2-element tuple with the data and the labels. The second element
            is None if the label is not specified.
        """
        df = pd.read_csv(self.data_path)
        
        # Select columns with the given prefixes
        selected_columns = [
            col
            for col in df.columns
            if any(prefix in col for prefix in self.feature_prefixes)
        ]
        data = df[selected_columns].to_numpy()

        # If features_as_channels is True, reshape the data to (N, C, T)
        # where N is the number of samples, C is the number of channels and
        # T is the number of time steps
        if self.features_as_channels:
            data = data.reshape(
                -1,
                len(self.feature_prefixes),
                data.shape[1] // len(self.feature_prefixes),
            )

        # Cast the data to the specified type
        if self.cast_to:
            data = data.astype(self.cast_to)

        # If label is specified, return the data and the labels
        if self.label:
            labels = df[self.label].to_numpy()
            return data, labels
        # If label is not specified, return only the data
        else:
            return data, None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if self.label:
            return self.data[index], self.labels[index]
        else:
            return self.data[index]
