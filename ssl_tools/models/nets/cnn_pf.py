from typing import Tuple
import torch
from torchmetrics import Accuracy

from ssl_tools.models.nets.simple import SimpleClassificationNet
from ssl_tools.models.utils import ZeroPadder2D


# Convolutional Neural Networks for Human Activity Recognition using Multiple
# Accelerometer and Gyroscope Sensors, from Ha, and Choi.
# https://ieeexplore.ieee.org/document/7727224


# (I1) 3 x 3, (C1) 2 x 3, (S1) 3 x 5, (C2) 2 x 3.
class CNN_PF_Backbone(torch.nn.Module):
    def __init__(
        self,
        pad_at: int,
        input_shape: Tuple[int, int, int],
        out_channels: int = 16,
        include_middle: bool = False,
    ):
        super().__init__()
        self.pad_at = pad_at
        self.input_shape = input_shape
        self.include_middle = include_middle
        self.out_channels = out_channels
        self.first_pad_size = 3 - 1  # kernel -1

        self.first_padder = ZeroPadder2D(
            pad_at=(pad_at,),
            padding_size=self.first_pad_size,
        )

        self.upper_part = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.input_shape[0],
                out_channels=self.out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(2, 3),
                stride=(2, 3),
                padding=1,
            ),
        )

        self.lower_part = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.input_shape[0],
                out_channels=self.out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(2, 3),
                stride=(2, 3),
                padding=1,
            ),
        )

        if self.include_middle:
            self.middle_part = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.input_shape[0],
                    out_channels=self.out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(
                    kernel_size=(2, 3),
                    stride=(2, 3),
                    padding=1,
                ),
            )

        self.shared_part = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=(
                    self.out_channels * 3
                    if self.include_middle
                    else self.out_channels * 2
                ),
                out_channels=64,
                kernel_size=(3, 5),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(2, 3),
                stride=(2, 3),
                padding=1,
            ),
        )

    def forward(self, x):
        # X = (batch_size, channels, sensors, time_steps)
        # X = (8, 1, 6, 60)

        # After pad: (8, 1, 8, 60)
        x = self.first_padder(x)

        # upper slice (8, 1, 5, 60)
        upper_x = x[:, :, : self.pad_at + self.first_pad_size, :]
        upper_x = self.upper_part(upper_x)
        zeros_1 = torch.zeros(
            upper_x.size(0),
            upper_x.size(1),
            3 - 1,
            upper_x.size(3),
            device=x.device,
        )

        upper_x = torch.cat(
            [upper_x, zeros_1],
            dim=2,
        )

        # lower slice (8, 1, 5, 60)
        lower_x = x[:, :, self.pad_at :, :]
        lower_x = self.lower_part(lower_x)
        zeros_2 = torch.zeros(
            lower_x.size(0),
            lower_x.size(1),
            3 - 1,
            lower_x.size(3),
            device=x.device,
        )

        lower_x = torch.cat(
            [zeros_2, lower_x],
            dim=2,
        )

        if self.include_middle:
            # x is already middle
            middle_x = self.middle_part(x)
            concatenated_x = torch.cat([upper_x, middle_x, lower_x], dim=1)

        else:
            concatenated_x = torch.cat([upper_x, lower_x], dim=1)

        result_x = self.shared_part(concatenated_x)
        return result_x


class CNN_PF_2D(SimpleClassificationNet):
    def __init__(
        self,
        pad_at: int,
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        out_channels: int = 16,
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        include_middle: bool = False,
    ):
        self.pad_at = pad_at
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.num_classes = num_classes

        backbone = CNN_PF_Backbone(
            pad_at=pad_at,
            input_shape=input_shape,
            out_channels=out_channels,
            include_middle=include_middle,
        )
        self.fc_input_channels = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = self._create_fc(self.fc_input_channels, num_classes)
        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            val_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
            test_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
        )

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int, int]
            The input shape of the network.

        Returns
        -------
        int
            The number of features after the convolutional layers.
        """
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=512, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )


class CNN_PFF_2D(CNN_PF_2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, include_middle=True)


# def test_cnn_pf_2d():
#     input_shape = (1, 6, 60)

#     data_module = RandomDataModule(
#         num_samples=8,
#         num_classes=6,
#         input_shape=input_shape,
#         batch_size=8,
#     )

#     model = CNN_PF_2D(pad_at=3, input_shape=input_shape)
#     print(model)

#     trainer = L.Trainer(
#         max_epochs=1, logger=False, devices=1, accelerator="cpu"
#     )

#     trainer.fit(model, datamodule=data_module)


# def test_cnn_pff_2d():
#     input_shape = (1, 6, 60)

#     data_module = RandomDataModule(
#         num_samples=8,
#         num_classes=6,
#         input_shape=input_shape,
#         batch_size=8,
#     )

#     model = CNN_PFF_2D(pad_at=3, input_shape=input_shape)
#     print(model)

#     trainer = L.Trainer(
#         max_epochs=1, logger=False, devices=1, accelerator="cpu"
#     )

#     trainer.fit(model, datamodule=data_module)


# if __name__ == "__main__":
#     import logging

#     logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
#     logging.getLogger("lightning").setLevel(logging.ERROR)
#     logging.getLogger("lightning.pytorch.core").setLevel(logging.ERROR)

#     # test_cnn_1d()
#     test_cnn_pf_2d()
#     test_cnn_pff_2d()
