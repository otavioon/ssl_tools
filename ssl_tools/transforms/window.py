import numpy as np
from numpy.lib.stride_tricks import as_strided

class Windowize:
    def __init__(self, time_segments: int = 15, stride: int = None):
        self.time_segments = time_segments
        self.stride = stride if stride is not None else time_segments
        assert self.time_segments > 0, "time_segments must be positive"
        assert self.stride > 0, "stride must be positive"
        assert self.stride <= self.time_segments, "stride must be less than or equal to time_segments"
        
    def __call__(self, x: np.ndarray):
        if x.shape[-1] < self.time_segments:
            raise ValueError(f"Input data length is less than time_segments: {x.shape[-1]} < {self.time_segments}")
        
        windowed_data = []
        for i in range(0, x.shape[-1] - self.time_segments + 1, self.stride):
            window = x[..., i:i+self.time_segments]
            windowed_data.append(window)
        windowed_data = np.stack(windowed_data, axis=-1)
        windowed_data = np.moveaxis(windowed_data, -1, -2)
        return windowed_data
    
   
# def main():
#     from ssl_tools.transforms.pad import ZeroPaddingBetween
#     from ssl_tools.transforms.utils import Squeeze, Unsqueeze
    
#     np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
#     input_data = resulted_data = np.random.rand(1, 6, 60)
#     print(f"Input data shape: {input_data.shape}")
#     print(input_data)
#     print("-" * 50)
    
#     squeezer = Squeeze(axis=0)
#     resulted_data = squeezer(input_data)
#     print(f"Resulted data shape (squeezer): {resulted_data.shape}")
#     print(resulted_data)
#     print("-" * 50)
    
    
#     # windowizer = Windowize(time_segments=60, stride=None)
#     # resulted_data = windowizer(input_data)
#     # print(f"Resulted data shape (windowizer): {resulted_data.shape}")
#     # print(resulted_data)
#     # print("-" * 50)
    
#     # resulted_data = resulted_data.reshape(-1, resulted_data.shape[-1])
#     # print(f"Resulted data shape (reshape): {resulted_data.shape}")
#     # print(resulted_data)
#     # print("-" * 50)
    
#     padder = ZeroPaddingBetween(pad_every=3, padding_size=2)
#     resulted_data  = padder(resulted_data)
#     print(f"Resulted data shape (padder): {resulted_data.shape}")
#     print(resulted_data)
#     print("-" * 50)
    
#     unsqueezer = Unsqueeze(axis=0)
#     resulted_data = unsqueezer(resulted_data)
#     print(f"Resulted data shape (unsqueezer): {resulted_data.shape}")
#     print(resulted_data)
#     print("-" * 50)
    
    
# if __name__ == "__main__":
#     main()
