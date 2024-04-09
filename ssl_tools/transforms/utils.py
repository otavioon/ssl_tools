from typing import List
import numpy as np

from librep.base import Transform


class Composer(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
        
    def transform(self, X):
        for t in self.transforms:
            X = t.transform(X)
        return X
            
    def __call__(self, X):
        return self.transform(X)
    
class Identity(Transform):
    def transform(self, X):
        return X
    
    def __call__(self, X):
        return self.transform(X)
    
class Reshape(Transform):
    def __init__(self, shape):
        self.shape = shape
        
    def transform(self, X):
        return X.reshape(self.shape)
    
    def __call__(self, X):
        return self.transform(X)
    
class Flatten(Transform):
    def transform(self, X):
        return X.reshape(X.shape[0], -1)

    def __call__(self, X):
        return self.transform(X)


class Squeeze(Transform):
    def __init__(self, axis=None):
        self.axis = axis
    
    def transform(self, X):
        return np.squeeze(X, axis=self.axis)

    def __call__(self, X):
        return self.transform(X)
    
    
    
class Unsqueeze(Transform):
    def __init__(self, axis):
        self.axis = axis
    
    def transform(self, X):
        return np.expand_dims(X, axis=self.axis)
    
    def __call__(self, X):
        return self.transform(X)
    
    
class Cast(Transform):
    def __init__(self, dtype):
        self.dtype = dtype
        
    def transform(self, X):
        return X.astype(self.dtype)
    
    def __call__(self, X):
        return self.transform(X)
    
class PerChannelTransform(Transform):
    def __init__(self, transform: Transform):
        self.transform = transform
        
    def transform(self, X):
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
        
        for i in range(X.shape[0]):
            datas.append(self.transform(X[i]))
            
        datas = np.stack(datas)
        return datas

    def __call__(self, X):
        return self.transform(X)


class StackComposer:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def transform(self, X):
        datas = []
        for t in self.transforms:
            data = t(X)
            datas.append(data)
            
        return np.stack(datas)

    
    def __call__(self, x):
        return self.transform(x)