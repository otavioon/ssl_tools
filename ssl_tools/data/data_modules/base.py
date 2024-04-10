import lightning as L
from torch.utils.data import DataLoader, Dataset

class SimpleDataModule(L.LightningDataModule):   
    def _load_dataset(self, split_name: str) -> Dataset:
        raise NotImplementedError

    def _get_loader(self, split_name: str, shuffle: bool) -> DataLoader:
        raise NotImplementedError
    
    def train_dataloader(self) -> DataLoader:
        return self._get_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_loader("validation", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_loader("predict", shuffle=False)
