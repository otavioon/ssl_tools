#!/usr/bin/env python3

import os
from typing import List, Tuple
import lightning as L
import torch

from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.nets import Simple1DConvNetwork
from ssl_tools.pipelines.cli import auto_main
from ssl_tools.pipelines.mlflow_train import (
    LightningTrainMLFlow,
    LightningFineTuneMLFlow,
)
from ssl_tools.pipelines.har_classification.utils import FFT, Spectrogram
from ssl_tools.models.ssl.classifier import SSLDiscriminator


class Simple1DConvNetTrain(LightningTrainMLFlow):
    MODEL = "Simple1DConvNet"

    def __init__(
        self,
        data: str | List[str],
        input_shape: Tuple[int, int] = (6, 60),
        num_classes: int = 6,
        transforms: str = "identity",
        num_workers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        self.transforms = None
        assert transforms in ["identity", "fft", "spectrogram"]
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True)]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram()]
        else:
            self.transforms = None

    def get_model(self) -> L.LightningModule:
        model = Simple1DConvNetwork(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
            transforms=self.transforms,
        )

        return data_module


class Simple1DConvNetFineTune(LightningFineTuneMLFlow):
    MODEL = "Simple1DConvNet"

    def __init__(
        self,
        data: str | List[str],
        num_classes: int = 6,
        transforms: str = "identity",
        num_workers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.num_classes = num_classes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        self.transforms = None
        assert transforms in ["identity", "fft", "spectrogram"]
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True)]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram()]
        else:
            self.transforms = None

    def get_model(self) -> L.LightningModule:
        model: Simple1DConvNetwork = self.load_model()
        model.fc = torch.nn.Identity()

        classifier = torch.nn.Linear(model.fc_input_channels, self.num_classes)
        model = SSLDiscriminator(
            backbone=model,
            head=classifier,
            loss_fn=torch.nn.CrossEntropyLoss(),
            update_backbone=self.update_backbone,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
            transforms=self.transforms,
        )

        return data_module


if __name__ == "__main__":
    options = {
        "train": Simple1DConvNetTrain,
        "finetune": Simple1DConvNetFineTune,
    }
    auto_main(options)
