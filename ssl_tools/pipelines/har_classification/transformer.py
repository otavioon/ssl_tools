#!/usr/bin/env python3

from typing import List, Tuple
import lightning as L
import torch

from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.nets.transformer import SimpleTransformer
from ssl_tools.pipelines.cli import auto_main
from ssl_tools.pipelines.mlflow_train import (
    LightningTrainMLFlow,
    LightningFineTuneMLFlow,
)
from ssl_tools.pipelines.har_classification.utils import (
    FFT,
    Spectrogram,
    DimensionAdder,
    SwapAxes,
)
import os
from ssl_tools.models.ssl.classifier import SSLDiscriminator


class SimpleTransformerTrain(LightningTrainMLFlow):
    MODEL = "Transformer"

    def __init__(
        self,
        data: str | List[str],
        in_channels: int = 6,
        dim_feedforward=60,
        num_classes: int = 6,
        heads: int = 1,
        num_layers: int = 1,
        num_workers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.in_channels = in_channels
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.heads = heads
        self.num_layers = num_layers
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

    def get_model(self) -> L.LightningModule:
        model = SimpleTransformer(
            in_channels=self.in_channels,
            dim_feedforward=self.dim_feedforward,
            num_classes=self.num_classes,
            heads=self.heads,
            num_layers=self.num_layers,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
            transforms=[SwapAxes(0, 1)],
        )

        return data_module


class SimpleTransformerFineTune(LightningFineTuneMLFlow):
    def __init__(
        self,
        data: str | List[str],
        num_classes: int = 6,
        num_workers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.num_classes = num_classes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

    def get_model(self) -> L.LightningModule:
        model: SimpleTransformer = self.load_model()
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
            transforms=[SwapAxes(0, 1)],
        )

        return data_module


if __name__ == "__main__":
    options = {
        "train": SimpleTransformerTrain,
        "finetune": SimpleTransformerFineTune,
    }
    auto_main(options)
