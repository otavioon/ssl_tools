import torch
import torch.nn as nn
import lightning as L
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from ssl_tools.models.nets.simple import SimpleClassificationNet


class SimpleTransformer(SimpleClassificationNet):
    def __init__(
        self,
        in_channels: int = 6,
        dim_feedforward=60,
        num_classes: int = 6,
        heads: int = 2,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
    ):
        self.in_channels = in_channels
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.heads = heads
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.fc_input_channels = in_channels * dim_feedforward

        super().__init__(
            backbone=TransformerEncoder(
                TransformerEncoderLayer(
                    in_channels,
                    dim_feedforward=dim_feedforward,
                    nhead=heads,
                    batch_first=True,
                ),
                num_layers=num_layers,
            ),
            fc=torch.nn.Sequential(
                torch.nn.Dropout(),
                torch.nn.Linear(self.fc_input_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(128, num_classes),
            ),
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            learning_rate=learning_rate,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# def main():
#     from torch.utils.data import DataLoader

#     class RandomDataset:
#         def __init__(
#             self,
#             num_samples: int = 64,
#             num_classes: int = 6,
#             input_shape: tuple = (6, 60),
#         ):
#             self.num_samples = num_samples
#             self.num_classes = num_classes
#             self.input_shape = input_shape

#         def __len__(self):
#             return self.num_samples

#         def __getitem__(self, idx):
#             return (
#                 torch.randn(*self.input_shape),
#                 torch.randint(0, self.num_classes, (1,)).item(),
#             )

#     class RandomDataModule(L.LightningDataModule):
#         def __init__(
#             self, num_samples, num_classes, input_shape, batch_size: int = 1
#         ):
#             super().__init__()
#             self.num_samples = num_samples
#             self.num_classes = num_classes
#             self.input_shape = input_shape
#             self.batch_size = batch_size

#         def train_dataloader(self):
#             return DataLoader(
#                 RandomDataset(
#                     self.num_samples, self.num_classes, self.input_shape
#                 ),
#                 batch_size=self.batch_size,
#             )

#     data_module = RandomDataModule(
#         num_samples=16, num_classes=6, input_shape=(60, 6), batch_size=1
#     )

#     model = SimpleTransformer()

#     trainer = L.Trainer(max_epochs=1, logger=False, accelerator="cpu")

#     trainer.fit(model, datamodule=data_module)


# if __name__ == "__main__":
#     main()
