#!/usr/bin/env python3

import os
import lightning as L
import torch

from ssl_tools.pipelines.cli import auto_main
from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.ssl.modules.heads import TFCPredictionHead
from ssl_tools.pipelines.har_classification.utils import PredictionHeadClassifier
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


class TFCHeadClassifierTrain(LightningTrainMLFlow):
    MODEL = "TFCPredictionHead"

    def __init__(
        self,
        data: str,
        input_size: int = 360,
        num_classes: int = 6,
        transforms: str = "identity",
        num_workers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.input_size = input_size
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
        model = PredictionHeadClassifier(
            prediction_head=TFCPredictionHead(
                input_dim=self.input_size,
                output_dim=self.num_classes,
            ),
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


class TFCHeadClassifierFineTune(LightningFineTuneMLFlow):
    MODEL = "TFCPredictionHead"

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
        model: PredictionHeadClassifier = self.load_model()
        model.fc = torch.nn.Identity()
        classifier = torch.nn.Linear(model.backbone.output_dim, self.num_classes)
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
        "train": TFCHeadClassifierTrain,
        "finetune": TFCHeadClassifierFineTune,
    }
    auto_main(options)