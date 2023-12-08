import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from librep.base import Transform


class FFT(Transform):
    def transform(self, sample: np.ndarray):
        return np.abs(np.fft.fft(sample, axis=1))
    
    def __call__(self, sample: np.ndarray):
        return self.transform(sample)


class AddRemoveFrequency(Transform):
    def __init__(self, add_pertub_ratio=0.1, remove_pertub_ratio=0.1):
        self.add_pertub_ratio = add_pertub_ratio
        self.remove_pertub_ratio = remove_pertub_ratio

    def add_frequency(self, sample: np.ndarray):
        mask = torch.FloatTensor(sample.shape).uniform_() > (
            1 - self.add_pertub_ratio
        ) # only pertub_ratio of all values are True
        max_amplitude = sample.max()
        random_am = torch.rand(mask.shape) * (max_amplitude * 0.1)
        pertub_matrix = mask * random_am
        return sample + pertub_matrix.numpy()

    def remove_frequency(self, sample: np.ndarray):
        mask = (
            torch.FloatTensor(sample.shape).uniform_()
            > self.remove_pertub_ratio
        ).numpy()  # maskout_ratio are False
        return sample * mask

    def transform(self, sample: np.ndarray):
        sample_1 = self.remove_frequency(sample)
        sample_2 = self.add_frequency(sample)
        augmented_sample = sample_1 + sample_2
        return augmented_sample


    def __call__(self, sample: np.ndarray):
        return self.transform(sample)