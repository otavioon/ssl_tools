from bisect import bisect_right

class ConcatDataset:
    """
    Concatenate multiple datasets1
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.slices = self._get_slices(datasets)
        
    @staticmethod
    def _get_slices(datasets):
        i = 0
        slices = []
        for d in datasets:
            i += len(d)
            slices.append(i)
        return slices
            
    def __getitem__(self, i):
        bucket = bisect_right(self.slices, i)
        if bucket >= len(self.datasets):
            raise IndexError("Index out of range")
        
        return self.datasets[bucket][i-self.slices[bucket]]
        

    def __len__(self):
        return self.slices[-1]