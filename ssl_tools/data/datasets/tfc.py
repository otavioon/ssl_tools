#!/usr/bin/env python
# coding: utf-8

from typing import List

import torch
from torch.utils.data import Dataset
from typing import Union, Callable


from librep.base import Transform
import numpy as np


class TFCDataset(Dataset):
    def __init__(
        self,
        data: Dataset,
        length_alignment: int = 178,
        time_transforms: Union[Callable, List[Callable]] = None,
        frequency_transforms: Union[Callable, List[Callable]] = None,
        cast_to: str = "float32",
        only_time_frequency: bool = False,
    ):
        """Time-Frequency Contrastive (TFC) Dataset. This dataset is intented
        to be used using TFC technique. Given a dataset with time-domain signal,
        this dataset will calculate the FFT of the signal and apply the
        specified transforms to the time and frequency domainof each sample.
        It will return a 5-element tuple with the following elements:
        1. The original time-domain signal
        2. The label of the signal
        3. Time augmented time-domain signal
        4. The frequency-domain signal
        5. The augmented frequency-domain signal

        Note that, if samples are 1-D arrays, the transforms will be applied
        directly to the data. If samples are 2-D arrays, the transforms will
        be applied to each channel separately.

        Parameters
        ----------
        data : Dataset
            A dataset with samples. The sample must be a tuple with the data
            and the label. The data must be a tensor of shape (C, T), where C
            is the number of channels and T is the number of time steps. If no
            channels are present, the data must be of shape (T,).
        length_alignment : int, optional
            Truncate the features to this value
        time_transforms : Union[Transform, List[Transform]], optional
            List of transforms to apply to the time domain.
        frequency_transforms : Union[Transform, List[Transform]], optional
            List of transforms to apply to the frequency domain
        cast_to : str, optional
            Cast the data to the given type, by default "float32"
        only_time_frequency : bool, optional
            If True, the data returned will be a 2-element tuple with the
            (time, frequency) as the first element (without augmentation) and 
            the label as the second element, by default False
            
        Examples
        --------
        >>> from ssl_tools.data.datasets import MultiModalSeriesCSVDataset
        >>> data_path = "data.csv"
        >>> dataset = MultiModalSeriesCSVDataset(
                data_path,
                feature_prefixes=["accel-x", "accel-y", "accel-z"],
                label="class"
            )
        >>> dataset = TFCDataset(dataset, length_alignment=180)
        >>> dataset[0]
        >>> (
            torch.Tensor([[-0.0001, -0.0001, -0.0001,  ..., -0.0001, -0.0001, -0.0001]]),   # time
            0,
            torch.Tensor([[-0.5020, -0.5020, -0.5020,  ..., -0.5020, -0.5020, -0.5020]]),   # time augmented
            torch.Tensor([[0.1, 0.1, 0.1,  ..., 0.1, 0.1, 0.00101]]),                       # frequency
            torch.Tensor([[-0.5020, -0.5020, -0.5020,  ..., -0.5020, -0.5020, -0.5020]]),   # frequency augmented
        )
        """
        self.dataset = data
        self.length_alignment = length_alignment
        self.cast_to = cast_to
        self.only_time_frequency = only_time_frequency

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

    class FFT:
        def __init__(self, absolute: bool = True):
            """Simple wrapper to apply the FFT to the data

            Parameters
            ----------
            absolute : bool, optional
                If True, returns the absolute value of FFT, by default True
            """
            self.absolute = absolute

        def __call__(self, x: np.ndarray) -> np.ndarray:
            """Apply the FFT to the data

            Parameters
            ----------
            x : np.ndarray
                A 1-D array with the data

            Returns
            -------
            np.ndarray
                The FFT of the data
            """
            result = np.fft.fft(x)
            if self.absolute:
                result = np.abs(result)
            return result

    def _apply_transforms(
        self, x: np.ndarray, transforms: List[Transform]
    ) -> np.ndarray:
        """Apply a list of transforms to the data

        Parameters
        ----------
        x : np.ndarray
            The 1-D array with the data
        transforms : List[Transform]
            A sequence of transforms to apply in the data

        Returns
        -------
        np.ndarray
            The transformed data
        """
        # Apply the transforms on the data, one by one
        for transform in transforms:
            x = transform(x)
        return x

    def _apply_transforms_per_axis(
        self, data: np.ndarray, transforms: List[Transform]
    ) -> np.ndarray:
        """Split the data into channels and apply the transforms to each channel
        separately.

        Parameters
        ----------
        data : np.ndarray
            The data to be transformed. It must be a 2-D array with the shape
            (C, T), where C is the number of channels and T is the number of
            time steps.
        transforms : List[Transform]
            A sequence of transforms to apply in the data

        Returns
        -------
        np.ndarray
            An 2-D array with the transformed data. The array has the number of
            channels as the first dimension.
        """
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

        # If data is a 1-D array, convert it to a 2-D array
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        data = data[:, :self.length_alignment]

        time_aug = self._apply_transforms_per_axis(
            data, self.aug_time_transforms
        )
        freq = self._apply_transforms_per_axis(
            data, self.frequency_transform
        )
        freq_aug = self._apply_transforms_per_axis(
            data, self.aug_frequency_transforms
        )

        # Cast the data to the specified type
        if self.cast_to:
            data = data.astype(self.cast_to)
            time_aug = time_aug.astype(self.cast_to)
            freq = freq.astype(self.cast_to)
            freq_aug = freq_aug.astype(self.cast_to)
            
        if self.only_time_frequency:
            return (data, freq), label

        # Returns the data, the label, the time augmented data, the frequency
        # data and the frequency augmented data
        return data, label, time_aug, freq, freq_aug
