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
    
class Identity:
    def transform(self, X):
        return X
    
class Reshape(Transform):
    def __init__(self, shape):
        self.shape = shape
        
    def transform(self, X):
        return X.reshape(self.shape)
    
class Flatten(Transform):
    def transform(self, X):
        return X.reshape(X.shape[0], -1)
