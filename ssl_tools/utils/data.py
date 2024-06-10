from bisect import bisect_right
from torch.utils.data import DataLoader
import lightning as L

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
    
    
def get_split_dataloader(
    stage: str, data_module: L.LightningDataModule
) -> DataLoader:
    if stage == "train":
        data_module.setup("fit")
        return data_module.train_dataloader()
    elif stage == "validation":
        data_module.setup("fit")
        return data_module.val_dataloader()
    elif stage == "test":
        data_module.setup("test")
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Invalid stage: {stage}")


def full_dataset_from_dataloader(dataloader: DataLoader):
    return dataloader.dataset[:]


def get_full_data_split(
    data_module: L.LightningDataModule,
    stage: str,
):
    dataloader = get_split_dataloader(stage, data_module)
    return full_dataset_from_dataloader(dataloader)

