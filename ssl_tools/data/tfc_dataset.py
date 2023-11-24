#!/usr/bin/env python
# coding: utf-8

from typing import List

import torch
from torch.utils.data import Dataset
from typing import Union


from librep.base import Transform


class TFCContrastiveDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor = None,
        length_alignment: int = 178,
        time_transforms: Union[Transform, List[Transform]] = None,
        frequency_transforms: Union[Transform, List[Transform]] = None,
    ):
        """
        A PyTorch dataset for time-frequency consistency.

        Args:
            data (torch.Tensor): The input data tensor of shape (num_samples, num_channels, num_features).
            labels (torch.Tensor, optional): The target labels tensor of shape (num_samples,). Defaults to None.
            length_alignment (int, optional): The length to which the input data is aligned. Defaults to 178.
            time_transforms (Union[Transform, List[Transform]], optional): A list of time-domain transforms to apply to the input data. Defaults to None.
            frequency_transforms (Union[Transform, List[Transform]], optional): A list of frequency-domain transforms to apply to the input data. Defaults to None.
        """
        assert len(data) == len(labels), "Data and labels must have the same length"

        self.data_time = data
        self.labels = labels
        self.length_alignment = length_alignment
        self.time_transforms = time_transforms or []
        self.frequency_transforms = frequency_transforms or []

        if not isinstance(self.time_transforms, list):
            self.time_transforms = [self.time_transforms]
        if not isinstance(self.frequency_transforms, list):
            self.frequency_transforms = [self.frequency_transforms]

        if len(self.data_time.shape) < 3:
            self.data_time = self.data_time.unsqueeze(2)

        if self.data_time.shape.index(min(self.data_time.shape)) != 1:
            self.data_time = self.data_time.permute(0, 2, 1)

        """Align the data to the same length, removing the extra features"""
        self.data_time = self.data_time[:, :1, : self.length_alignment]

        """Calculcate the FFT of the data and apply the transforms (if any)"""
        self.data_freq = torch.fft.fft(self.data_time).abs()

        # This could be done in the __getitem__ method
        # For now, we do it here to be more similar to the original implementation
        self.data_time_augmented = self.apply_transforms(
            self.data_time, self.time_transforms
        )
        self.data_freq_augmented = self.apply_transforms(
            self.data_freq, self.frequency_transforms
        )

    def apply_transforms(
        self, x: torch.Tensor, transforms: List[Transform]
    ) -> torch.Tensor:
        for transform in transforms:
            x = transform.fit_transform(x)
        return x

    def __len__(self):
        return len(self.data_time)

    def __getitem__(self, index):
        # Time processing
        return (
            self.data_time[index].float(),
            self.labels[index],
            self.data_time_augmented[index].float(),
            self.data_freq[index].float(),
            self.data_freq_augmented[index].float(),
        )
