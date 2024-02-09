from typing import Dict, List, Optional, Tuple, Union
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
from scipy.stats import zscore
from sklearn.cluster import KMeans
import pandas as pd
from functools import partial


# ----------------- Losses -----------------
class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6, *args, **kwargs):
        super().__init__()
        self.mse = torch.nn.MSELoss(*args, **kwargs)
        self.eps = eps

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return loss


# ----------------- Static anomaly threshold functions  -----------------
def mean_absolute_error(X, X_recon):
    if X.ndim > 2:
        return np.mean(np.abs(X - X_recon), axis=(1, 2))
    else:
        return np.mean(np.abs(X - X_recon), axis=1)


def mean_squared_error(X, X_recon):
    if X.ndim > 2:
        return np.mean(np.square(X - X_recon), axis=(1, 2))
    else:
        return np.mean(np.square(X - X_recon), axis=1)
    # return np.mean(np.square(X - X_recon), axis=(1, 2))


def root_mean_squared_error(X, X_recon):
    if X.ndim > 2:
        return np.sqrt(np.mean(np.square(X - X_recon), axis=(1, 2)))
    else:
        return np.sqrt(np.mean(np.square(X - X_recon), axis=1))
    # return np.sqrt(np.mean(np.square(X - X_recon), axis=(1, 2)))


def zscore_threshold_max(X_recon):
    return np.max(zscore(X_recon))


def zscore_threshold_std(X_recon, std):
    zscores = zscore(X_recon)
    return np.mean(zscores) + std * np.std(zscores)


def kmeans_threshold(X_recon, n_clusters=1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
        X_recon.reshape(-1, 1)
    )
    clusters = kmeans.predict(X_recon.reshape(-1, 1))
    cluster_distances = np.min(
        np.abs(kmeans.cluster_centers_ - X_recon), axis=1
    )
    return cluster_distances[0]


def sigma_threshold(X_recon, sigma):
    return np.mean(X_recon) + sigma * np.std(X_recon)


# def stats_max(X_recon):
#     stats = pd.DataFrame(X_recon).describe()
#     the_max = stats.filter(like="max", axis=0)
#     return float(the_max[0].iloc[0])


# ----------------- Experiments  -----------------


class CovidAnomalyDetectionTrain(LightningTrain):
    def __init__(
        self,
        data: str,
        input_shape: Tuple[int, ...],
        participant_ids: Optional[Union[int, List[int]]] = None,
        validation_split: float = 0.1,
        augment: bool = False,
        feature_column_prefix: str = "RHR",
        target_column: str = "anomaly",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.input_shape = input_shape
        self.participant_ids = participant_ids
        self.validation_split = validation_split
        self.augment = augment
        self.feature_column_prefix = feature_column_prefix
        self.target_column = target_column

    def _get_transforms(self):
        return [
            Identity(),
            Scale(),
            Rotate(),
            Permutate(),
            MagnitudeWarp(),
            TimeWarp(),
            WindowSlice(),
            WindowWarp(),
        ]

    def get_data_module(self) -> CovidUserAnomalyDataModule:
        return CovidUserAnomalyDataModule(
            data_path=self.data,
            participants=self.participant_ids,
            feature_column_prefix=self.feature_column_prefix,
            target_column=self.target_column,
            reshape=self.input_shape,
            train_transforms=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            validation_split=self.validation_split,
            dataset_transforms=self._get_transforms() if self.augment else None,
        )

    def get_model(self) -> L.LightningModule:
        raise NotImplementedError


class CovidAnomalyDetectionEvaluator(LightningTest):
    def __init__(
        self,
        train_data: str,
        test_data: str,
        train_participant: int,
        test_participant: int,
        input_shape: Tuple[int, ...],
        feature_column_prefix: str = "RHR",
        target_column: str = "anomaly",
        include_recovered_in_test: bool = False,
        results_dir: str = "results",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_data = train_data
        self.test_data = test_data
        self.input_shape = input_shape
        self.train_participant = train_participant
        self.test_participant = test_participant
        self.feature_column_prefix = feature_column_prefix
        self.target_column = target_column
        self.include_recovered_in_test = include_recovered_in_test
        self.results_dir = self.experiment_dir / results_dir
        self._error_funcs = {
            "mse": torch.nn.MSELoss(reduction="none"),
            "mae": torch.nn.L1Loss(reduction="none"),
            "rmse": RMSELoss(reduction="none"),
        }
        self._threshold_funcs = {
            "percentile_99": partial(np.percentile, q=99),
            "percentile_98": partial(np.percentile, q=98),
            "percentile_95": partial(np.percentile, q=95),
            "percentile_90": partial(np.percentile, q=90),
            "percentile_85": partial(np.percentile, q=85),
            "percentile_80": partial(np.percentile, q=80),
            "percentile_75": partial(np.percentile, q=75),
            "percentile_70": partial(np.percentile, q=70),
            "percentile_60": partial(np.percentile, q=60),
            "percentile_50": partial(np.percentile, q=50),
            "max": np.max,
            # "stats_max": stats_max,
            "mean": np.mean,
            "zscore_max": zscore_threshold_max,
            "zscore_std_1": partial(zscore_threshold_std, std=1),
            "zscore_std_2": partial(zscore_threshold_std, std=2),
            "zscore_std_3": partial(zscore_threshold_std, std=3),
            "kmeans_1": partial(kmeans_threshold, n_clusters=1),
            "sigma_2": partial(sigma_threshold, sigma=2),
            "sigma_3": partial(sigma_threshold, sigma=3),
        }

    def get_data_module(self) -> CovidUserAnomalyDataModule:
        return [
            CovidUserAnomalyDataModule(
                data_path=self.train_data,
                participants=self.train_participant,
                feature_column_prefix=self.feature_column_prefix,
                target_column=self.target_column,
                reshape=self.input_shape,
                train_transforms=None,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                dataset_transforms=None,
                include_recovered_in_test=self.include_recovered_in_test,
                discard_last_batch=False,
                shuffle_train=False,
            ),
            CovidUserAnomalyDataModule(
                data_path=self.test_data,
                participants=self.test_participant,
                feature_column_prefix=self.feature_column_prefix,
                target_column=self.target_column,
                reshape=self.input_shape,
                train_transforms=None,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                dataset_transforms=None,
                include_recovered_in_test=self.include_recovered_in_test,
                discard_last_batch=False,
                shuffle_train=False,
            ),
        ]

    def get_model(self) -> L.LightningModule:
        raise NotImplementedError

    def _calc_static_anomaly_thresholds(
        self, losses: np.ndarray
    ) -> Dict[str, float]:
        return {
            threshold_func_name: threshold_func(losses)
            for threshold_func_name, threshold_func in self._threshold_funcs.items()
        }

    def run_model(
        self,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ):
        train_datamodule, test_datamodule = data_module

        # Train
        train_datamodule.setup("fit")
        train_loader = train_datamodule.train_dataloader()
        x_hat_train = trainer.predict(model, train_loader)
        x_hat_train = torch.cat(x_hat_train)
        x_train = list(x for x, y in train_loader)
        x_train = torch.cat(x_train)
        x_hat_train = x_hat_train.view(x_hat_train.size(0), -1)
        x_train = x_train.view(x_hat_train.size(0), -1)

        # Test
        test_datamodule.setup("test")
        test_dataloader = test_datamodule.test_dataloader()
        x_hat_test = trainer.predict(model, test_dataloader)
        x_hat_test = torch.cat(x_hat_test)
        x_test = list(x for x, y in test_dataloader)
        x_test = torch.cat(x_test)
        x_hat_test = x_hat_test.view(x_hat_test.size(0), -1)
        x_test = x_test.view(x_hat_test.size(0), -1)
        y_test = list(y for x, y in test_dataloader)
        y_test = torch.cat(y_test)
        y_test = y_test.int().numpy()

        self.results_dir.mkdir(parents=True, exist_ok=True)
        reports = []

        # Iterate over losses
        for loss_name, loss_func in self._error_funcs.items():
            # Calculate the losses for the train set
            train_losses = loss_func(x_hat_train, x_train).numpy()
            train_losses = train_losses.mean(axis=1)

            test_losses = loss_func(x_hat_test, x_test).numpy()
            test_losses = test_losses.mean(axis=1)

            losses_df = pd.DataFrame({"loss": train_losses})
            losses_df.to_csv(
                self.results_dir / f"{loss_name}_train_losses.csv", index=False
            )

            # Calculate anomaly scores based on the train losses
            anomaly_thresholds = self._calc_static_anomaly_thresholds(
                train_losses
            )

            # Iterate over threshold functions
            for (
                threshold_func_name,
                threshold_value,
            ) in anomaly_thresholds.items():
                y_hat = test_losses > threshold_value

                y_hat = y_hat.astype(int)
                y_test = y_test.astype(int)

                report = classification_report(y_test, y_hat)
                report.update(
                    {
                        "loss_func": loss_name,
                        "threshold_func": threshold_func_name,
                        "threshold_value": threshold_value,
                    }
                )
                reports.append(report)

                predictions_df = pd.DataFrame(
                    {
                        "y_true": y_test,
                        "y_pred": y_hat,
                        "loss": test_losses,
                    }
                )
                predictions_df.to_csv(
                    self.results_dir
                    / f"{loss_name}_{threshold_func_name}_predictions.csv",
                    index=False,
                )
                # print(
                #     f"Predictions saved to {self.results_dir / f'{loss_name}_{threshold_func_name}_predictions.csv'}"
                # )

        report = pd.DataFrame(reports)
        print(
            report[
                [
                    "f1",
                    "uar",
                    "balanced_accuracy",
                    "loss_func",
                    "threshold_func",
                    "threshold_value",
                ]
            ]
            .sort_values(by="uar", ascending=False)
            .reset_index(drop=True)
            .to_markdown()
        )

        report.to_csv(self.results_dir / "results.csv", index=False)
        print(f"Results saved to {self.results_dir / 'results.csv'}")

        return report
