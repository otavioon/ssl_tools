from typing import List
from librep.base import Transform


class ContrastiveDataset:
    def __init__(self, X, transforms: List[Transform]):
        self.X = X
        self.transforms = transforms
        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]

    def __getitem__(self, index):
        data = self.X[index]
        return [t.transform(data) for t in self.transforms]

    def __len__(self):
        return len(self.X)