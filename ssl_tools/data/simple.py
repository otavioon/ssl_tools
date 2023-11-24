import numpy as np

class SimpleDataset:
    def __init__(self, X, y=None, cast_to: str = "float32"):
        self.X = X
        self.y = y
        self.cast_to = cast_to
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.cast_to is not None:
            x = x.astype(self.cast_to)
            
        if self.y is None:
            return x
        
        y = self.y[idx]
        if self.cast_to is not None:
            y = y.astype(self.cast_to)
            
        return x, y
