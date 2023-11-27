import numpy as np
from pathlib import Path


import torch

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Tuple
import zarr

class SimpleDataset:
    def __init__(self, X, y=None, cast_to: str = "float32"):
        self.X = X
        self.y = y
        self.cast_to = cast_to
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.cast_to is not None:
            x = x.astype(self.cast_to)
            
        if self.y is None:
            return x
        
        y = self.y[idx]
        if self.cast_to is not None:
            y = y.astype(self.cast_to)
            
        return x, y

class SimpleArrayDataset(Dataset):

    """
    Dataset creation
    Parameters:
    data_path: Path to the zarr file with the seismic data
    label_path: Path to the zarr file with the seismic attribute
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data_shape: tuple = (1, 500, 500),
        unsqueeze_dim: int = None,
        indexes: list = None,
    ):
        self.data = x
        self.label = y
        self.data_shape = data_shape
        self.unsqueeze_dim = unsqueeze_dim
        self.mirror_dim = []
        self.indexes = self._get_indexes() if indexes is None else indexes

    def _get_indexes(self):
        indexes = []
        range_i = self.data.shape[0] - self.data_shape[0]
        range_j = self.data.shape[1] - self.data_shape[1]
        range_k = self.data.shape[2] - self.data_shape[2]

        if range_i < 1:
            range_i = 1
        if range_j < 1:
            range_j = 1
        if range_k < 1:
            range_k = 1

        for i in range(0, range_i, self.data_shape[0]):
            for j in range(0, range_j, self.data_shape[1]):
                for k in range(0, range_k, self.data_shape[2]):
                    indexes.append((i, j, k))
        return indexes

    def __len__(self):
        return len(self.indexes)

    def _adjust_data(self, data, data_shape):
        """
        Dado o formato alvo, preenche data com 0s para garantir o formato desejado
        """
        diferencas = np.asarray(data_shape) - np.asarray(data.shape)
        return np.pad(
            data,
            ((0, diferencas[0]), (0, diferencas[1]), (0, diferencas[2])),
            "constant",
            constant_values=0,
        )

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        x0, y0, z0 = self.indexes[index]

        # ------------- Acquire data and label ----------------
        data = self.data[
            x0 : x0 + min(self.data_shape[0], self.data.shape[0]),
            y0 : y0 + min(self.data_shape[1], self.data.shape[1]),
            z0 : z0 + min(self.data_shape[2], self.data.shape[2]),
        ]
        label = self.label[
            x0 : x0 + min(self.data_shape[0], self.data.shape[0]),
            y0 : y0 + min(self.data_shape[1], self.data.shape[1]),
            z0 : z0 + min(self.data_shape[2], self.data.shape[2]),
        ]

        # ------------- Format data and label, padding it -------
        data = self._adjust_data(data, self.data_shape)
        label = self._adjust_data(label, self.data_shape)


        # ------------- Transform to torch ternsor ----------------
        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)

        # ------------- Do squeeze to data and label ----------------
        if self.unsqueeze_dim is not None:
            data = data.unsqueeze(0)
            label = label.unsqueeze(0)


        return (data, label)

    def __str__(self) -> str:
        return f"ArrayDataset with {len(self)} samples of shape {self.data_shape}"

    def __repr__(self) -> str:
        return str(self)


class ZarrArrayDataset(SimpleArrayDataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        label_path: Union[str, Path],
        data_shape: tuple = (1, 500, 500),
        unsqueeze_dim: int = None,
        indexes: list = None,
    ):
        self.data_path = data_path
        self.label_path = label_path
        x = self._load_data()
        y = self._load_label()

        super().__init__(
            x=x,
            y=y,
            data_shape=data_shape,
            unsqueeze_dim=unsqueeze_dim,
            indexes=indexes,
        )

    def _load_data(self):
        data = zarr.open(self.data_path, mode="r")
        return data

    def _load_label(self):
        label = zarr.open(self.label_path, mode="r")
        return label
