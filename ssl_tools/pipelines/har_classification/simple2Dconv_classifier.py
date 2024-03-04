#!/usr/bin/env python3

from typing import Tuple
import lightning as L
import torch

from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.nets import Simple2DConvNetwork
from ssl_tools.pipelines.cli import auto_main
from ssl_tools.pipelines.mlflow_train import (
    LightningTrainMLFlow,
    LightningFineTuneMLFlow,
)
from ssl_tools.pipelines.har_classification.utils import (
    FFT,
    Spectrogram,
    DimensionAdder
)
import os
from ssl_tools.models.ssl.classifier import SSLDiscriminator


class Simple2DConvNetTrain(LightningTrainMLFlow):
    MODEL = "Simple2DConvNet"

    def __init__(
        self,
        data: str,
        input_shape: Tuple[int, int, int] = (6, 1, 60),
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
            self.transforms = [FFT(absolute=True, centered=True), DimensionAdder(dim=1)]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram(), DimensionAdder(dim=1)]
        else:
            self.transforms = [DimensionAdder(dim=1)]
        

    def get_model(self) -> L.LightningModule:
        model = Simple2DConvNetwork(
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
            transforms=self.transforms
        )

        return data_module
    

class Simple2DConvNetFineTune(LightningFineTuneMLFlow):
    MODEL = "Simple2DConvNet"

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
        assert transforms in ["identity", "fft", "spectrogram"]
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True), DimensionAdder(dim=1)]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram(), DimensionAdder(dim=1)]
        else:
            self.transforms = [DimensionAdder(dim=1)]

    def get_model(self) -> L.LightningModule:
        model: Simple2DConvNetwork = self.load_model()
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
        "train": Simple2DConvNetTrain,
        "finetune": Simple2DConvNetFineTune

    }
    auto_main(options)
