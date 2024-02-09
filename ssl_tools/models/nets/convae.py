from typing import Tuple
import torch
from torch import nn
import torch.nn as nn


from ssl_tools.models.nets.simple import SimpleReconstructionNet
from ssl_tools.losses.contrastive_loss import ContrastiveLoss


class _ConvolutionalAutoEncoder(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int, int] = (1, 16)):
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_shape[0], 32, kernel_size=7, stride=2, padding=3
        )
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=7, stride=2, padding=3)
        self.conv_transpose1 = nn.ConvTranspose1d(
            16, 16, kernel_size=7, stride=2, padding=3, output_padding=1
        )
        self.dropout2 = nn.Dropout(0.2)
        self.conv_transpose2 = nn.ConvTranspose1d(
            16, 32, kernel_size=7, stride=2, padding=3, output_padding=1
        )
        self.conv_transpose3 = nn.ConvTranspose1d(
            32, input_shape[0], kernel_size=7, padding=3
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv_transpose1(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        return x
    

class _ConvolutionalAutoEncoder2D(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 4, 4)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=1)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, input_shape[0], kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.upsample1(x)
        x = torch.relu(self.conv4(x))
        x = self.upsample2(x)
        x = torch.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x


class ConvolutionalAutoEncoder(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (1, 16),
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            backbone=_ConvolutionalAutoEncoder(input_shape=input_shape),
            learning_rate=learning_rate,
            loss_fn=nn.MSELoss(),
        )
        self.input_shape = input_shape


class ConvolutionalAutoEncoder2D(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 4, 4),
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            backbone=_ConvolutionalAutoEncoder2D(input_shape=input_shape),
            learning_rate=learning_rate,
            loss_fn=nn.MSELoss(),
        )
        self.input_shape = input_shape


class ContrastiveConvolutionalAutoEncoder(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (1, 16),
        learning_rate: float = 1e-3,
        margin: float = 1.0,
    ):
        super().__init__(
            backbone=_ConvolutionalAutoEncoder(input_shape=input_shape),
            learning_rate=learning_rate,
            loss_fn=ContrastiveLoss(margin),
        )

class ContrastiveConvolutionalAutoEncoder2D(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (4, 4, 1),
        learning_rate: float = 1e-3,
        margin: float = 1.0,
    ):
        super().__init__(
            backbone=_ConvolutionalAutoEncoder2D(input_shape=input_shape),
            learning_rate=learning_rate,
            loss_fn=ContrastiveLoss(margin),
        )
        self.input_shape = input_shape