import numpy as np
from scipy.ndimage import gaussian_filter1d

from librep.base import Transform

   
class Scale(Transform):
    def __init__(self, mean: float = 1.0, sigma: float = 0.5):
        self.mean = mean
        self.sigma = sigma
        
    def transform(self, sample: np.ndarray):
        num_channels, num_time_steps = sample.shape
        
        if self.mean is None:
            self.mean = sample.mean()
        
        # Generate scaling factors for each channel and time step
        scaling_factors = np.random.normal(loc=self.mean, scale=self.sigma, size=(num_channels, num_time_steps))
        
        # Rescale each channel separately
        data_scaled = np.multiply(sample, scaling_factors[:, :])
        return data_scaled
    
    def __call__(self, sample: np.ndarray):
        return self.transform(sample)
    
    
class AddGaussianNoise(Transform):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
        
    def transform(self, sample: np.ndarray):
        noise = np.random.normal(self.mean, self.std, size=sample.shape)
        noisy_sample = sample + noise
        return noisy_sample
    
    def __call__(self, sample: np.ndarray):
        return self.transform(sample)
    
class Rotate(Transform):
    def transform(self, dataset: np.ndarray):
        flip = np.random.choice([-1, 1], size=(dataset.shape[0],dataset.shape[-1]))
        rotate_axis = np.arange(dataset.shape[-1])
        np.random.shuffle(rotate_axis)
        data_rotation = flip[:,:] * dataset[:,rotate_axis]
        return data_rotation
    
    def __call__(self, sample: np.ndarray):
        return self.transform(sample)
    
class LeftToRightFlip(Transform):
    def transform(self, sample: np.ndarray):
        return np.flip(sample, axis=-1)
    
    def __call__(self, sample: np.ndarray):
        return self.transform(sample)
    
class MagnitudeWrap(Transform):
    def __init__(self, max_magnitude=1.0):
        self.max_magnitude = max_magnitude
        
    def transform(self, sample: np.ndarray):
        magnitudes = np.linalg.norm(sample, axis=1)
        scaling_factors = self.max_magnitude / magnitudes
        scaled_sample = sample * scaling_factors[:, np.newaxis]
        return scaled_sample
    
    def __call__(self, sample: np.ndarray):
        return self.transform(sample)
    
class TimeAmplitudeModulation(Transform):
    def __init__(self, modulation_factor=0.1):
        self.modulation_factor = modulation_factor
        
    def transform(self, sample: np.ndarray):
        _, num_time_steps = sample.shape
        
        # Generate modulation factors for each time step
        modulation_factors = np.random.uniform(1 - self.modulation_factor, 1 + self.modulation_factor, size=num_time_steps)
        
        # Apply modulation to each time step
        modulated_sample = sample * modulation_factors[ np.newaxis, :]
        return modulated_sample

    def __call__(self, sample: np.ndarray):
        return self.transform(sample)
class RandomSmoothing(Transform):
    def __init__(self, sigma_range=(1, 1)):
        self.sigma_range = sigma_range
        
    def transform(self, sample: np.ndarray):
        num_channels, num_time_steps = sample.shape
        
        # Generate a random smoothing factor (sigma) for Gaussian filter
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        
        # Apply Gaussian smoothing along the time axis for each channel
        smoothed_sample = np.empty_like(sample)
        for channel in range(num_channels):
            smoothed_sample[channel, :] = gaussian_filter1d(sample[channel, :], sigma)
        
        return smoothed_sample
    
    def __call__(self, sample: np.ndarray):
        return self.transform(sample)