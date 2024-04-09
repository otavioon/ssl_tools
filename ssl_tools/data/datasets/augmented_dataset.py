from torch.utils.data import Dataset
from typing import Dict, List, Callable, Union
import itertools


class AugmentedDataset(Dataset):
    """Note: this class assumes that dataset is a Dataset object, and that
    the __getitem__ method of the dataset returns a tuple of n elements.
    """

    def __init__(
        self, dataset: Dataset, transforms: List[Union[Callable, Dict[int, Callable]]]
    ):
        """_summary_

        Parameters
        ----------
        dataset : Dataset
            _description_
        transforms : Dict[int, Callable]
            As each element (result of __getitem__) of the dataset is a
            n-element tuple, the transforms are applied to the n-th element
            of the tuple. The key of the dictionary is the index of the
            element of the tuple to apply the transform (0-indexed), and the
            value is the transform to apply.
        """
        self.dataset = dataset
        self.transforms = transforms
        
        if isinstance(transforms[0], Callable):
            self.transforms = [{0: t} for t in transforms]

        self.indexes = list(
            itertools.product(range(len(dataset)), range(len(transforms)))
        )

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        dataset_idx, transform_idx = self.indexes[idx]
        result = self.dataset[dataset_idx]
        result = list(result)
    
        for i, t in self.transforms[transform_idx].items():
            result[i] = t(result[i])       
        
        return tuple(result)
