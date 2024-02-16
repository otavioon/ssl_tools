#!/usr/bin/env python3

import lightning as L

from ssl_tools.experiments import LightningTrain, auto_main
from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.nets.simple import SimpleClassificationNet
from ssl_tools.models.layers.gru import GRUEncoder
from ssl_tools.experiments.har_classification._classification_base import (
    EvaluatorBase,
)
from ssl_tools.experiments.har_classification.utils import (
    FFT,
    Flatten,
    Spectrogram,
)
import torch

class GRUClassifier(SimpleClassificationNet):
    def __init__(
        self,
        hidden_size: int = 100,
        in_channels: int = 6,
        num_classes: int = 6,
        encoding_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        encoder = GRUEncoder(
            hidden_size=hidden_size,
            in_channels=in_channels,
            encoding_size=encoding_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        super().__init__(
            backbone=encoder,
            fc=torch.nn.Linear(encoding_size, num_classes),
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
        )

class GRUClassifierTrain(LightningTrain):
    _MODEL_NAME = "GRU"

    def __init__(
        self,
        data: str,
        hidden_size: int = 100,
        in_channels: int = 6,
        num_classes: int = 6,
        encoding_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        transforms: str = "identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoding_size = encoding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.transforms = None
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True)]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram()]
        else:
            self.transforms = None

    def get_model(self) -> L.LightningModule:
        model = GRUClassifier(
            hidden_size=self.hidden_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            encoding_size=self.encoding_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
            transforms=self.transforms
        )

        return data_module


class GRUClassifierTest(EvaluatorBase):
    _MODEL_NAME = "GRU"

    def __init__(
        self,
        data: str,
        hidden_size: int = 100,
        in_channels: int = 6,
        num_classes: int = 6,
        encoding_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        transforms: str = "identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoding_size = encoding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.transforms = None
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True)]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram()]
        else:
            self.transforms = None

    def get_model(self) -> L.LightningModule:
        model = GRUClassifier(
            hidden_size=self.hidden_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            encoding_size=self.encoding_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
            transforms=self.transforms
        )
        return data_module


if __name__ == "__main__":
    options = {
        "fit": GRUClassifierTrain,
        "test": GRUClassifierTest,
    }
    auto_main(options)
