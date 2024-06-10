from typing import List
import torch
from torch.utils.data import DataLoader
from typing import List
import torch
import lightning as L

class ShapePrinter(torch.nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"-- ({self.name}) Shape: {x.shape}")
        return x


class ZeroPadder2D(torch.nn.Module):
    def __init__(self, pad_at: List[int], padding_size: int):
        super().__init__()
        self.pad_at = pad_at
        self.padding_size = padding_size

    def forward(self, x):
        # X = (Batch, channels, H, W)
        # X = (8, 1, 6, 60)

        for i in self.pad_at:
            left = x[:, :, :i, :]
            zeros = torch.zeros(
                x.shape[0], x.shape[1], self.padding_size, x.shape[3],
                device=x.device
            )
            right = x[:, :, i:, :]

            x = torch.cat([left, zeros, right], dim=2)
            # print(f"-- Left.shape: {left.shape}")
            # print(f"-- Zeros.shape: {zeros.shape}")
            # print(f"-- Right.shape: {right.shape}")
            # print(f"-- X.shape: {x.shape}")
        
        return x
    
    def __str__(self) -> str:
        return f"ZeroPadder2D(pad_at={self.pad_at}, padding_size={self.padding_size})"
    
    def __repr__(self) -> str:
        return str(self)
    
    
class RandomDataset:
    def __init__(
        self,
        num_samples: int = 64,
        num_classes: int = 6,
        input_shape: tuple = (6, 60),
        transforms: list = None,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.transforms = transforms or []
        assert isinstance(
            self.transforms, list
        ), "transforms must be a list"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.randn(*self.input_shape)
        label = torch.randint(0, self.num_classes, (1,)).item()

        for t in self.transforms:
            data = t(data)

        return data, label

class RandomDataModule(L.LightningDataModule):
    def __init__(
        self,
        num_samples,
        num_classes,
        input_shape,
        transforms: list = None,
        batch_size: int = 1,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.transforms = transforms
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            RandomDataset(
                self.num_samples,
                self.num_classes,
                self.input_shape,
                transforms=self.transforms,
            ),
            batch_size=self.batch_size,
        )
