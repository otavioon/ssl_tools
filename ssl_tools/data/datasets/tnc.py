from typing import Tuple
import numpy as np

from torch.utils.data import Dataset
from statsmodels.tsa.stattools import adfuller
import math



class TNCDataset(Dataset):
    def __init__(
        self,
        data: Dataset,
        window_size: int,
        mc_sample_size: int = 20,
        significance_level: float = 0.01,
        repeat: int = 1,  # Simply repeat the vecvor 'augmentation' times
        cast_to: str = "float32",
    ):
        """Temporal Neighbourhood Coding (TNC) dataset. This dataset is used
        to pre-train self-supervised models. The dataset obtain close and
        distant samples from a time series.
        
        The dataset returns a 3-element tuple with the following elements:
        1. W_t, a window centered at random time step t, with window_size. It
            is a numpy array with shape (n_features, window_size).
        2. X_p, a set of close samples. It is a numpy array with shape
            (mc_sample_size, n_features, window_size).
        3. X_n, a set of distant samples. It is a numpy array with shape
            (mc_sample_size, n_features, window_size).
        Note that the number of distant samples may be less than mc_sample_size.

        Parameters
        ----------
        data : Dataset
            A dataset with samples. Each sample of the dataset must be a numpy
            array of shape (n_features, time_steps). In the case of HAR, where
            we have tri-axial accelerometer and gyroscope data, the shape of
            each sample should be (6, time_steps). the time_steps may vary
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
            Simple repeat the element of the dataset ``repeat`` times
        cast_to : str, optional
            Cast the data to the given type, by default "float32"
        """
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.significance_level = significance_level
        assert isinstance(repeat, int), "Repeat must be an integer"
        self.repeat = repeat
        self.cast_to = cast_to

    def __len__(self) -> int:
        return len(self.data) * self.repeat

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a sample from the dataset. The sample is a tuple with 3
        elements, to know:
        1. W_t, a window centered at random time step t, with window_size. It
            is a numpy array with shape (n_features, window_size).
        2. X_p, a set of close samples. It is a numpy array with shape
            (mc_sample_size, n_features, window_size).
        3. X_n, a set of distant samples. It is a numpy array with shape
            (mc_sample_size, n_features, window_size).
        Note that the number of distant samples may be less than mc_sample_size.

        Parameters
        ----------
        idx : int
            Index of the sample to select the close and distant samples.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple with 3 elements (W_t, X_p, X_n).
        """
        # To repeat augmentation
        idx = idx % len(self.data)
        # Get the data.
        data = self.data[idx]
        time_len = data.shape[-1]

        # Select a random time step t. The value must be within the range
        # [2*window_size, time_len - 2*window_size]. We do this to assure
        # that we have enough data from left and right of the window
        t = np.random.randint(
            2 * self.window_size, time_len - 2 * self.window_size
        )
        # The sample is a window centered at t, with window_size / 2 elements
        # before and after t (X[t - δ, t + δ]])
        x_t = data[:, t - self.window_size // 2 : t + self.window_size // 2]

        # Find the close samples and the delta. X_close is a numpy array with
        # shape (mc_sample_size, n_features, window_size).
        X_close, delta = self._find_neighours(data, t)

        # Find the distant samples. X_distant is a numpy array with shape
        # (mc_sample_size, n_features, window_size). Note that the number of
        # distant samples may be less than mc_sample_size.
        X_distant = self._find_non_neighours(data, t, delta)

        # Cast the data to the given type
        if self.cast_to is not None:
            x_t = x_t.astype(self.cast_to)
            X_close = X_close.astype(self.cast_to)
            X_distant = X_distant.astype(self.cast_to)

        # Returns the sample, a 3-element tuple
        return x_t, X_close, X_distant

    def _find_neighours(
        self, data: np.ndarray, t: int
    ) -> Tuple[np.ndarray, float]:
        """Given a time series ``x_t`` and a time step ``t``, find the close
        samples and the delta. The close samples are selected using the ADF
        test. The delta adjusts the neighbourhood size.

        Parameters
        ----------
        data : np.ndarray
            The time series
        t : int
            The time step to find the close samples

        Returns
        -------
        Tuple[np.ndarray, float]
            A 2-element tuple. The first element is a numpy array with the
            shape (mc_sample_size, n_features, window_size). The second element
            is the delta, used to adjust the neighbourhood size.
        """
        # Get the length of the time series
        num_features, time_len = data.shape

        # ---- Do the ADF test ----
        corr = []  # List of average p-values

        # Iterate over 3 different window sizes (W_t). The window sizes are
        # [window_size, 2*window_size, 3*window_size].
        for wsize in range(
            self.window_size, 4 * self.window_size, self.window_size
        ):
            # For each window of ``wsize`` centered at ``t``, we perform the
            # ADF test on each feature, and store the average p-value, in the
            # list ``corr``.
            try:
                # Used to store the sum of p-values calculated for each feature.
                p_val_sum = 0
                # Iterate over the features and calculate the adf test
                for feat_idx in range(num_features):
                    # Get the a window of size wsize centered at t of the
                    # current feature. Note that if t - wsize < 0 or t +
                    # wsize > time_len, the window will be truncated. Thus,
                    # it is possible that the window is not centered at t.
                    window_sample = np.array(
                        data[
                            feat_idx,
                            max(0, t - wsize) : min(time_len, t + wsize),
                        ]
                    )

                    # Reshape the data to be a 1D array
                    window_sample = window_sample.reshape(-1)

                    # Execute the ADF test on data. The ADF test returns a
                    # tuple with 5 values. The second value (index 1) is the
                    # p-value.
                    p_val = adfuller(np.array(window_sample))[1]
                    # Add the p-value to the sum. If the p-value is NaN, add
                    # ``self.significance_level`` (value used to accept the
                    # null hypothesis)
                    p_val_sum += (
                        self.significance_level if math.isnan(p_val) else p_val
                    )

                # Average the p-values and append it to the list of p-values
                p_val_avg = p_val_sum / num_features
                corr.append(p_val_avg)
            except:
                corr.append(0.6)

        # Check how many p-values are greater than the significance level, it
        # is, how many p-values accept the null hypothesis (the series is not
        # stationary).
        epsilon = (
            len(corr)
            if len(np.where(np.array(corr) >= self.significance_level)[0]) == 0
            else (np.where(np.array(corr) >= self.significance_level)[0][0] + 1)
        )

        # Calculate the new delta to, it is, adjust the neighbourhood size
        delta = 5 * epsilon * self.window_size

        # Select ``self.mc_sample_size`` random time steps from the
        # neighbourhood (``window_size * epsilon``) of ``t`` (only after ``t``).
        # The close samples will be centered at these time steps.
        t_p = [
            int(t + np.random.randn() * epsilon * self.window_size)
            for _ in range(self.mc_sample_size)
        ]

        # Iterate over each time step in ``t_p`` and adjust its value to
        # assure that the window will not be out of bounds (truncated)
        t_p = [
            max(
                self.window_size // 2 + 1,
                min(t_pp, time_len - self.window_size // 2),
            )
            for t_pp in t_p
        ]

        # Get the windows centered at the time steps in ``t_p``. The windows
        # will have size ``window_size``.
        centered_windows = [
            data[
                :,
                t_ind - self.window_size // 2 : t_ind + self.window_size // 2,
            ]
            for t_ind in t_p
        ]

        # Convert the list of windows to a numpy array
        x_p = np.stack(centered_windows)

        # Returns the close samples and the delta
        return x_p, delta

    def _find_non_neighours(
        self, data: np.ndarray, t: int, delta: float = 0.0
    ) -> np.ndarray:
        """Find distant samples. The samples will be selected from the
        neighbourhood of ``t``. If ``t`` is a time stemp from the first half
        of the time series, the neighbourhood will be the second half of the
        time series. Otherwise, the neighbourhood will be the first half of
        the time series.

        Parameters
        ----------
        data : np.ndarray
            The time series
        t : int
            The time step to find the close samples
        delta : float, optional
            Factor used to adjust the neighbourhood size, by default 0.0

        Returns
        -------
        np.ndarray
            An array with the distant samples. The shape of the array is
            (mc_sample_size, n_features, window_size). Note that if the list
            of distant samples is empty, mc_sample_size will be 1.
        """
        # Get the length of the time series
        num_features, time_len = data.shape

        # Check if t is greater than time_len / 2. If it is, the distant
        # samples will be selected from the left side of the time series.
        # Otherwise, the distant samples will be selected from the right side.

        # Select random time steps. ``t_n`` is a list with
        # ``self.mc_sample_size`` random time steps from the neighbourhood of
        # ``t``. If ``t`` is greater than ``time_len / 2``, the neighbourhood
        # will be the first half of the time series. Otherwise, the
        # neighbourhood will be the second half of the time series.
        if t > time_len / 2:
            t_n = np.random.randint(
                self.window_size // 2,
                max((t - delta + 1), self.window_size // 2 + 1),
                self.mc_sample_size,
            )
        else:
            t_n = np.random.randint(
                min((t + delta), (time_len - self.window_size - 1)),
                (time_len - self.window_size // 2),
                self.mc_sample_size,
            )

        # Get the windows centered at the time steps in ``t_p``. The windows
        # will have size ``window_size``.
        x_n = np.stack(
            [
                data[
                    :,
                    t_ind
                    - self.window_size // 2 : t_ind
                    + self.window_size // 2,
                ]
                for t_ind in t_n
            ]
        )

        # If the list of distant samples is empty, select a random window from
        # the time series. If ``t`` is greater than ``time_len / 2``, the
        # window will be selected from the first half of the time series.
        if len(x_n) == 0:
            rand_t = np.random.randint(0, self.window_size // 5)
            if t > time_len / 2:
                x_n = data[:, rand_t : rand_t + self.window_size].unsqueeze(0)
            else:
                x_n = data[
                    :, time_len - rand_t - self.window_size : time_len - rand_t
                ].unsqueeze(0)
        # Returns the distant samples
        return x_n
