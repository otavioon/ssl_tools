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



# TODO: A way of removing the need to add the path to the root of
# the project
import lightning as L
import torch


from ssl_tools.models.ssl.tnc import build_tnc, TNC
from ssl_tools.data.data_modules import (
    TNCHARDataModule,
    MultiModalHARSeriesDataModule,
)
from torchmetrics import Accuracy
from ssl_tools.models.ssl.classifier import SSLDiscriminator
from ssl_tools.models.ssl.modules.heads import TNCPredictionHead

import lightning as L
import torch

from torchmetrics import Accuracy
from ssl_tools.models.ssl.classifier import SSLDiscriminator



class TNCPreTrain(LightningTrainMLFlow):
    def __init__(
        self,
        data: str,
        encoding_size: int = 10,
        in_channel: int = 6,
        window_size: int = 60,
        mc_sample_size: int = 20,
        w: float = 0.05,
        significance_level: float = 0.01,
        repeat: int = 5,
        pad_length: bool = True,
        num_classes: int = 6,
        num_workers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.encoding_size = encoding_size
        self.in_channel = in_channel
        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.w = w
        self.significance_level = significance_level
        self.repeat = repeat
        self.pad_length = pad_length
        self.num_classes = num_classes
        self.num_classes = num_classes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )

    def get_model(self) -> L.LightningModule:
        model = build_tnc(
            encoding_size=self.encoding_size,
            in_channel=self.in_channel,
            mc_sample_size=self.mc_sample_size,
            w=self.w,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = TNCHARDataModule(
            self.data,
            pad=self.pad_length,
            window_size=self.window_size,
            mc_sample_size=self.mc_sample_size,
            significance_level=self.significance_level,
            repeat=self.repeat,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return data_module
    
    
class TNCFineTune(LightningFineTuneMLFlow):
    def __init__(self, data: str, num_classes: int = 6, num_workers: int = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.num_classes = num_classes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        
    def get_model(self) -> L.LightningModule:
        model: TNC = self.load_model()
        classifier = TNCPredictionHead(
            input_dim=model.discriminator.input_size,
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
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
        )

        return data_module

if __name__ == "__main__":
    options = {
        "train": TNCPreTrain,
        "finetune": TNCFineTune
    }
    auto_main(options)
