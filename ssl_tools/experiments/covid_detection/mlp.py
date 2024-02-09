#!/usr/bin/env python3

from ssl_tools.experiments.covid_detection.classification_base import (
    CovidDetectionTrain,
    CovidDetectionEvaluator,
)
import lightning as L
from ssl_tools.experiments import auto_main
from ssl_tools.models.nets.simple import MLPClassifier
import torch

class MLPClassifierTrain(CovidDetectionTrain):
    _MODEL_NAME = "mlp"

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 128,
        num_hidden_layers: int = 1,
        num_classes: int = 1,
        learning_rate: float = 1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(reshape=(input_size, ), *args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def get_model(self) -> L.LightningModule:
        return MLPClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            output_size=self.num_classes,
            learning_rate=self.learning_rate,
            loss_fn=torch.nn.BCELoss(),
        )


class MLPClassifierTest(CovidDetectionEvaluator):
    _MODEL_NAME = "mlp"

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 128,
        num_hidden_layers: int = 1,
        num_classes: int = 1,
        learning_rate: float = 1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        

    def get_model(self) -> L.LightningModule:
        return MLPClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            output_size=self.num_classes,
            learning_rate=self.learning_rate,
        )


if __name__ == "__main__":
    options = {
        "fit": MLPClassifierTrain,
        "test": MLPClassifierTest,
    }
    auto_main(options, print_args=True)
