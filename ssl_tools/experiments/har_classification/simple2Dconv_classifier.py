#!/usr/bin/env python3

from typing import Tuple
import lightning as L

from ssl_tools.experiments import LightningTrain, auto_main
from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.nets import Simple2DConvNetwork
from ssl_tools.experiments.har_classification._classification_base import EvaluatorBase
import numpy as np

from ssl_tools.experiments.har_classification.utils import (
    FFT,
    Flatten,
    Spectrogram,
    DimensionAdder
)



        

class Simple2DConvNetTrain(LightningTrain):
    _MODEL_NAME = "Simple2DConvNet"

    def __init__(
        self,
        data: str,
        input_shape: Tuple[int, int, int] = (6, 1, 60),
        num_classes: int = 6,
        transforms: str = "identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.input_shape = input_shape
        self.num_classes = num_classes
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


class Simple2DConvNetTest(EvaluatorBase):
    _MODEL_NAME = "Simple2DConvNet"

    def __init__(
        self,
        data: str,
        input_shape: Tuple[int, int, int] = (6, 1, 60),
        num_classes: int = 6,
        transforms: str = "identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.input_shape = input_shape
        self.num_classes = num_classes
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
    
if __name__ == "__main__":
    options = {
        "fit": Simple2DConvNetTrain,
        "test": Simple2DConvNetTest,
    }
    auto_main(options)
