#!/usr/bin/env python3

from ssl_tools.experiments.covid_detection.anomaly_detection_base import (
    CovidAnomalyDetectionTrain,
    CovidAnomalyDetectionEvaluator,
)
import lightning as L
from ssl_tools.experiments import auto_main
from ssl_tools.models.nets.convae import ConvolutionalAutoEncoder2D




class ConvolutionalAutoencoder2DAnomalyDetectionTrain(CovidAnomalyDetectionTrain):
    _MODEL_NAME = "cae2d"

    def get_model(self) -> L.LightningModule:
        model =  ConvolutionalAutoEncoder2D(
            input_shape=self.input_shape,
            learning_rate=self.learning_rate,
        )
        print(model)
        return model


class ConvolutionalAutoencoder2DAnomalyDetectionTest(
    CovidAnomalyDetectionEvaluator
):
    _MODEL_NAME = "cae2d"

    def get_model(self) -> L.LightningModule:
        model = ConvolutionalAutoEncoder2D(
            input_shape=self.input_shape,
        )
        return model


if __name__ == "__main__":
    options = {
        "fit": ConvolutionalAutoencoder2DAnomalyDetectionTrain,
        "test": ConvolutionalAutoencoder2DAnomalyDetectionTest,
    }
    auto_main(options, print_args=True)
