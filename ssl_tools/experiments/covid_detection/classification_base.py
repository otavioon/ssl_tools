from typing import List, Optional, Tuple, Union
import lightning as L
from ssl_tools.experiments import LightningTrain, LightningTest
from ssl_tools.experiments.covid_detection.classfication_report import (
    classification_report,
)
from ssl_tools.data.data_modules import CovidUserAnomalyDataModule
from ssl_tools.transforms.time_1d_full import (
    Identity,
    Scale,
    Rotate,
    Permutate,
    MagnitudeWarp,
    TimeWarp,
    WindowSlice,
    WindowWarp,
)


import lightning as L
import torch
import numpy as np


from typing import List

import numpy as np

import torchmetrics
import pandas as pd


class CovidDetectionTrain(LightningTrain):
    def __init__(
        self,
        data: str,
        reshape: Optional[Tuple[int, ...]] = None,
        validation_split: float = 0.1,
        balance: bool = False,
        feature_column_prefix: str = "RHR",
        target_column: str = "anomaly",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.reshape = reshape
        self.validation_split = validation_split
        self.balance = balance
        self.feature_column_prefix = feature_column_prefix
        self.target_column = target_column

    def get_data_module(self) -> CovidUserAnomalyDataModule:
        return CovidUserAnomalyDataModule(
            data_path=self.data,
            participants=None,
            feature_column_prefix=self.feature_column_prefix,
            target_column=self.target_column,
            reshape=self.reshape,
            train_transforms=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            validation_split=self.validation_split,
            dataset_transforms=None,
            balance=self.balance,
            train_baseline_only=False
        )


class CovidDetectionEvaluator(LightningTest):
    def __init__(
        self,
        data: str,
        feature_column_prefix: str = "RHR",
        target_column: str = "anomaly",
        results_file: str = "results.csv",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.feature_column_prefix = feature_column_prefix
        self.target_column = target_column
        self.results_file = self.experiment_dir / results_file

    def get_data_module(self) -> CovidUserAnomalyDataModule:
        return CovidUserAnomalyDataModule(
            data_path=self.data,
            participants=None,
            feature_column_prefix=self.feature_column_prefix,
            target_column=self.target_column,
            reshape=None,
            train_transforms=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            train_baseline_only=False
        )

    def run_model(
        self,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ):
        y_hat = trainer.predict(model, datamodule=data_module)
        y_hat = torch.cat(y_hat)
        y = list(y for x, y in trainer.predict_dataloaders)
        y = torch.cat(y)
        n_classes = len(torch.unique(y))

        results = {
            "accuracy": [
                torchmetrics.functional.accuracy(
                    y_hat.view(-1),
                    y.view(-1),
                    num_classes=n_classes,
                    task="binary",
                ).item()
            ],
            "f1": [
                torchmetrics.functional.f1_score(
                    y_hat.view(-1),
                    y.view(-1),
                    num_classes=n_classes,
                    task="binary",
                ).item()
            ],
            "roc_auc": [
                torchmetrics.functional.auroc(
                    y_hat.view(-1),
                    y.view(-1),
                    num_classes=n_classes,
                    task="binary",
                ).item()
            ],
        }

        results = pd.DataFrame(results)
        print(results.to_markdown())

        results.to_csv(self.results_file, index=False)
        print(f"Results saved to {self.results_file}")
        return results
