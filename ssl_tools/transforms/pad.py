import numpy as np

class ZeroPaddingBetween:
    def __init__(self, pad_every: int = 3, padding_size: int = 2, discard_last: bool = True):
        self.pad_every = pad_every
        self.padding_size = padding_size
        self.discard_last = discard_last
        
    def __call__(self, x: np.ndarray):
        data = []
        time_steps = x.shape[-1]
        
        for i in range(len(x)):
            data.append(x[i])
            
            # Do not pad last
            if i == len(x)-1 and self.discard_last:
                continue
            
            if (i+1) % self.pad_every == 0:
                zeros = np.zeros((self.padding_size, time_steps))
                data.append(zeros)
        return np.vstack(data)