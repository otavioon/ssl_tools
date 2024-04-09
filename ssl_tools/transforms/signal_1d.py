import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import torch

from librep.base import Transform


# class FFT(Transform):
#     def transform(self, sample: np.ndarray):
#         return np.abs(np.fft.fft(sample, axis=1))
    
#     def __call__(self, sample: np.ndarray):
#         return self.transform(sample)


class FFT:
    def __init__(self, absolute: bool = True):
        """Simple wrapper to apply the FFT to the data

        Parameters
        ----------
        absolute : bool, optional
            If True, returns the absolute value of FFT, by default True
        """
        self.absolute = absolute

    def transform(self, x: np.ndarray) -> np.ndarray:
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
    
    def __call__(self, sample: np.ndarray):
        return self.transform(sample)
    
    
class WelchPowerSpectralDensity:
    def __init__(self, fs: int = 1/20, nperseg: int = None, noverlap: int = None, return_onesided=False, absolute: bool = True):
        """Simple wrapper to apply the Welch Power Spectral Density to the data

        Parameters
        ----------
        fs : int, optional
            The sampling frequency, by default 20
        nperseg : int, optional
            The number of data points in each segment, by default 30
        noverlap : int, optional
            The number of points of overlap between segments, by default 15
        return_onesided : bool, optional
            If True, return the one-sided PSD, by default False
        absolute : bool, optional
            If True, returns the absolute value of PSD, by default True
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.return_onesided = return_onesided
        self.absolute = absolute
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the Welch Power Spectral Density to the data

        Parameters
        ----------
        x : np.ndarray
            A 1-D array with the data

        Returns
        -------
        np.ndarray
            The Welch Power Spectral Density of the data
        """
        _, Pxx = signal.welch(x, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, return_onesided=self.return_onesided)
        if self.absolute:
            Pxx = np.abs(Pxx)
        
        return Pxx
    
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