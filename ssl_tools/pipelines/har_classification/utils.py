from typing import Dict
import numpy as np
from scipy import fftpack
from scipy.signal import spectrogram
import torch
import torchmetrics

from ssl_tools.models.nets.simple import SimpleClassificationNet


class FFT:
    def __init__(self, absolute: bool = True, centered: bool = False):
        self.absolute = absolute
        self.centered = centered

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Aplly FFT to the input signal. It apply the FFT into each channel
        of the input signal.

        Parameters
        ----------
        x : np.ndarray
            An array with shape (n_channels, n_samples) containing the input

        Returns
        -------
        np.ndarray
            The FFT of the input signal. The shape of the output is
            (n_channels, n_samples) if absolute is False, and
            (n_channels, n_samples//2) if absolute is True.
        """

        datas = []
        for data in x:
            fft_data = fftpack.fft(data)
            if self.absolute:
                fft_data = np.abs(fft_data)
            if self.centered:
                fft_data = fft_data[: len(fft_data) // 2]
            datas.append(fft_data)

        return np.array(datas)


class Spectrogram:
    def __init__(self, fs=20, nperseg=16, noverlap=8, nfft=16):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Aplly Spectrogram to the input signal. It apply the Spectrogram into each channel
        of the input signal.

        Parameters
        ----------
        x : np.ndarray
            An array with shape (n_channels, n_samples) containing the input

        Returns
        -------
        np.ndarray
            The Spectrogram of the input signal. The shape of the output is
            (n_channels, n_samples) if absolute is False, and
            (n_channels, n_samples//2) if absolute is True.
        """

        datas = []
        for data in x:
            f, t, Sxx = spectrogram(
                data,
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
            )
            Sxx = np.log(Sxx + 1e-10)
            datas.append(Sxx)

        return np.array(datas)


class Flatten:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Flatten the input signal. It apply the flatten into each channel
        of the input signal.

        Parameters
        ----------
        x : np.ndarray
            An array with shape (n_channels, n_samples) containing the input

        Returns
        -------
        np.ndarray
            The flatten of the input signal. The shape of the output is
            (n_channels, n_samples).
        """

        return x.reshape(-1)


class DimensionAdder:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x):
        x = np.expand_dims(x, axis=self.dim)
        return x


class PredictionHeadClassifier(SimpleClassificationNet):
    def __init__(self, prediction_head: torch.nn.Module, num_classes: int = 6):
        super().__init__(
            backbone=prediction_head,
            fc=torch.nn.Identity(),
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            val_metrics={
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
            },
            test_metrics={
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
            },
        )

class SwapAxes:
    def __init__(self, axis1: int, axis2: int):
        self.axis1 = axis1
        self.axis2 = axis2

    def __call__(self, x):
        x = np.swapaxes(x, self.axis1, self.axis2)
        return x