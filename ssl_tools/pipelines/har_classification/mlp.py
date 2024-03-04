#!/usr/bin/env python3

import os
import lightning as L
import torch

from ssl_tools.pipelines.cli import auto_main
from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.nets import MLPClassifier
from ssl_tools.pipelines.mlflow_train import (
    LightningTrainMLFlow,
    LightningFineTuneMLFlow,
)
from ssl_tools.pipelines.har_classification.utils import (
    FFT,
    Flatten,
    Spectrogram,
)
from ssl_tools.models.ssl.classifier import SSLDiscriminator


class MLPClassifierTrain(LightningTrainMLFlow):
    def __init__(
        self,
        data: str,
        input_size: int = 360,
        hidden_size: int = 64,
        num_hidden_layers: int = 1,
        num_classes: int = 6,
        transforms: str = "identity",
        num_workers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

        self.transforms = None
        assert transforms in ["identity", "flatten", "fft", "spectrogram"]
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True), Flatten()]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram(), Flatten()]
        else:
            self.transforms = [Flatten()]

    def get_model(self) -> L.LightningModule:
        model = MLPClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            output_size=self.num_classes,
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


class MLPClassifierFineTune(LightningFineTuneMLFlow):
    def __init__(
        self,
        data: str,
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
        assert transforms in ["identity", "flatten", "fft", "spectrogram"]
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True), Flatten()]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram(), Flatten()]
        else:
            self.transforms = [Flatten()]

    def get_model(self) -> L.LightningModule:
        model: MLPClassifier = self.load_model()
        model.fc = torch.nn.Identity()
        classifier = torch.nn.Linear(model.hidden_size, self.num_classes)
        
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
        "train": MLPClassifierTrain,
        "finetune": MLPClassifierFineTune,
    }
    auto_main(options)
