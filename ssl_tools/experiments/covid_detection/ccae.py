#!/usr/bin/env python3

from ssl_tools.experiments.covid_detection.anomaly_detection_base import (
    CovidAnomalyDetectionTrain,
    CovidAnomalyDetectionEvaluator,
)
import lightning as L
from ssl_tools.experiments import auto_main
from ssl_tools.models.nets.convae import ContrastiveConvolutionalAutoEncoder




class ConvolutionalAutoencoderAnomalyDetectionTrain(CovidAnomalyDetectionTrain):
    _MODEL_NAME = "ccae"

    def get_model(self) -> L.LightningModule:
        return ContrastiveConvolutionalAutoEncoder(
            input_shape=self.input_shape,
            learning_rate=self.learning_rate,
        )


class ConvolutionalAutoencoderAnomalyDetectionTest(
    CovidAnomalyDetectionEvaluator
):
    _MODEL_NAME = "ccae"

    def get_model(self) -> L.LightningModule:
        assert self.input_shape is not None, "input_shape must be specified"
        assert len(self.input_shape) == 2, "input_shape must be of length 2"

        model = ContrastiveConvolutionalAutoEncoder(
            input_shape=self.input_shape,
        )
        return model


if __name__ == "__main__":
    options = {
        "fit": ConvolutionalAutoencoderAnomalyDetectionTrain,
        "test": ConvolutionalAutoencoderAnomalyDetectionTest,
    }
    auto_main(options, print_args=True)
