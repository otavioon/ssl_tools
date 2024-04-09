from typing import Dict, Tuple
import torch
import lightning as L
from torchmetrics import Accuracy

from ssl_tools.models.nets.simple import SimpleClassificationNet


class Simple1DConvNetwork(SimpleClassificationNet):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        """Create a simple 1D Convolutional Network with 3 layers and 2 fully
        connected layers.

        Parameters
        ----------
        input_shape : Tuple[int, int], optional
            A 2-tuple containing the number of input channels and the number of
            features, by default (6, 60).
        num_classes : int, optional
            Number of output classes, by default 6
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 1e-3
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_channels=input_shape[0])
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

    def _create_backbone(self, input_channels: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 64, 5),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv1d(64, 64, 5),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv1d(64, 64, 5),
            torch.nn.ReLU(),
        )

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int]
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
            torch.nn.Dropout(),
            torch.nn.Linear(input_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, num_classes),
        )

class Simple2DConvNetwork(SimpleClassificationNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (6, 1, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        """Create a simple 2D Convolutional Network with 3 layers and 2 fully
        connected layers.

        Parameters
        ----------
        input_shape : Tuple[int, int, int], optional
            A 3-tuple containing the number of input channels, and the number of
            the 2D input shape, by default (6, 1, 60).
        num_classes : int, optional
            Number of output classes, by default 6
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 1e-3
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_channels=input_shape[0])
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

    def _create_backbone(self, input_channels: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            # First 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=(1, input_channels),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            # Second 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, input_channels)
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1, 2), stride=2),
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
            torch.nn.Linear(in_features=input_features, out_features=1000),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1000, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=num_classes),
        )


    # def forward(self, x):
    #     x = x.permute(1, 0, 2)
    #     return super().forward(x)