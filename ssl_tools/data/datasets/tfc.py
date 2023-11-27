#!/usr/bin/env python
# coding: utf-8

from typing import List

import torch
from torch.utils.data import Dataset
from typing import Union


from librep.base import Transform
import numpy as np


class TFCDataset(Dataset):
    def __init__(
        self,
        data: Dataset,
        length_alignment: int = 178,
        time_transforms: Union[Transform, List[Transform]] = None,
        frequency_transforms: Union[Transform, List[Transform]] = None,
        cast_to: str = "float32",
    ):
        """Time-Frequency Contrastive (TFC) Dataset. This dataset is intented
        to be used using TFC technique. Given a time-domain signal, the
        dataset will calculate the FFT of the signal and apply the specified
        transforms to the time and frequency domain.
        It will return a 5-element tuple with the following elements:
        - The original time-domain signal
        - The label of the signal
        - Time augmented signal
        - The frequency signal
        - The frequency augmented signal


        Parameters
        ----------
        data : Dataset
            A dataset with samples. The sample must be a tuple with the data
            and the label. The data must be a tensor of shape (C, T), where C
            is the number of channels and T is the number of time steps. If no
            channels are present, the data must be of shape (T,).
        length_alignment : int, optional
            _description_, by default 178
        time_transforms : Union[Transform, List[Transform]], optional
            List of transforms to apply to the time domain,
        frequency_transforms : Union[Transform, List[Transform]], optional
            List of transforms to apply to the frequency domain
        cast_to : str, optional
            Cast the data to the given type, by default "float32"
        """
        self.dataset = data
        self.length_alignment = length_alignment
        self.cast_to = cast_to

        # Augmented time transforms
        self.aug_time_transforms = time_transforms or []
        if not isinstance(self.aug_time_transforms, list):
            self.aug_time_transforms = [self.aug_time_transforms]

        # Frequcnecy transforms
        self.frequency_transform = [self.FFT(absolute=True)]

        # Augmented frequency transforms
        self.aug_frequency_transforms = frequency_transforms or []
        if not isinstance(self.aug_frequency_transforms, list):
            self.aug_frequency_transforms = [self.aug_frequency_transforms]

        # We insert the FFT as the first transform and then the augmentations
        self.aug_frequency_transforms = [
            self.FFT(absolute=True)
        ] + self.aug_frequency_transforms

        # This could be done in the __getitem__ method
        # For now, we do it here to be more similar to the original implementation
        # self.data_time_augmented = self.apply_transforms(
        #     self.data_time, self.time_transforms
        # )
        # self.data_freq_augmented = self.apply_transforms(
        #     self.data_freq, self.frequency_transforms
        # )

    class FFT(Transform):
        def __init__(self, absolute: bool = True):
            self.absolute = absolute

        def transform(self, x: np.ndarray) -> np.ndarray:
            result = np.fft.fft(x)
            if self.absolute:
                result = np.abs(result)
            return result

    def _apply_transforms(
        self, x: torch.Tensor, transforms: List[Transform]
    ) -> torch.Tensor:
        for transform in transforms:
            x = transform.fit_transform(x)
        return x

    def _apply_transforms_per_axis(
        self, data: np.ndarray, transforms: List[Transform]
    ):
        """Apply the transforms to each axis of the data"""
        datas = []

        for i in range(data.shape[0]):
            result = self._apply_transforms(data[i], transforms)
            datas.append(result)

        datas = np.stack(datas)
        return datas

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, label = self.dataset[index]

        # Align the data to the same length, removing the extra features
        if data.ndim == 1:
            data = data[: self.length_alignment]
        else:
            data = data[:, : self.length_alignment]

        if data.ndim == 1:
            time_aug = self._apply_transforms(data, self.aug_time_transforms)
            freq = self._apply_transforms(data, self.frequency_transform)
            freq_aug = self._apply_transforms(
                data, self.aug_frequency_transforms
            )
        else:
            time_aug = self._apply_transforms_per_axis(
                data, self.aug_time_transforms
            )
            freq = self._apply_transforms_per_axis(
                data, self.frequency_transform
            )
            freq_aug = self._apply_transforms_per_axis(
                data, self.aug_frequency_transforms
            )

        if self.cast_to:
            data = data.astype(self.cast_to)
            time_aug = time_aug.astype(self.cast_to)
            freq = freq.astype(self.cast_to)
            freq_aug = freq_aug.astype(self.cast_to)

        # Time processing
        return data, label, time_aug, freq, freq_aug
