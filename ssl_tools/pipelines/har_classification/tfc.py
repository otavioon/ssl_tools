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

#!/usr/bin/env python3

import lightning as L
import torch

from torchmetrics import Accuracy
from ssl_tools.models.ssl.classifier import SSLDiscriminator
from ssl_tools.models.ssl.modules.heads import TFCPredictionHead
from ssl_tools.models.ssl.tfc import build_tfc_transformer, TFC
from ssl_tools.data.data_modules import TFCDataModule


class TFCTrain(LightningTrainMLFlow):
    def __init__(
        self,
        data: str,
        label: str = "standard activity code",
        encoding_size: int = 128,
        in_channels: int = 6,
        length_alignment: int = 60,
        use_cosine_similarity: bool = True,
        temperature: float = 0.5,
        features_as_channels: bool = True,
        jitter_ratio: float = 2,
        num_classes: int = 6,
        num_workers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.label = label
        self.encoding_size = encoding_size
        self.in_channels = in_channels
        self.length_alignment = length_alignment
        self.use_cosine_similarity = use_cosine_similarity
        self.temperature = temperature
        self.features_as_channels = features_as_channels
        self.jitter_ratio = jitter_ratio
        self.num_classes = num_classes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

    def get_model(self) -> L.LightningModule:
        model = build_tfc_transformer(
            encoding_size=self.encoding_size,
            in_channels=self.in_channels,
            length_alignment=self.length_alignment,
            use_cosine_similarity=self.use_cosine_similarity,
            temperature=self.temperature,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = TFCDataModule(
            self.data,
            batch_size=self.batch_size,
            label=self.label,
            features_as_channels=self.features_as_channels,
            length_alignment=self.length_alignment,
            jitter_ratio=self.jitter_ratio,
            num_workers=self.num_workers,
            only_time_frequency=False,
        )
        return data_module


class TFCFineTune(LightningFineTuneMLFlow):
    def __init__(
        self,
        data: str,
        num_classes: int = 6,
        num_workers: int = None,
        length_alignment: int = 60,
        encoding_size: int = 128,
        features_as_channels: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.num_classes = num_classes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.length_alignment = length_alignment
        self.encoding_size = encoding_size
        self.features_as_channels = features_as_channels

    def get_model(self) -> L.LightningModule:
        model: TFC = self.load_model()
        classifier = TFCPredictionHead(
            input_dim=2 * self.encoding_size,
            output_dim=self.num_classes,
        )
        model = SSLDiscriminator(
            backbone=model,
            head=classifier,
            loss_fn=torch.nn.CrossEntropyLoss(),
            update_backbone=self.update_backbone,
        )
        return model


    def get_data_module(self) -> L.LightningDataModule:
        data_module = TFCDataModule(
            self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=self.features_as_channels,
            length_alignment=self.length_alignment,
            num_workers=self.num_workers,
            only_time_frequency=True,
        )
        return data_module


if __name__ == "__main__":
    options = {"train": TFCTrain, "finetune": TFCFineTune}
    auto_main(options)
