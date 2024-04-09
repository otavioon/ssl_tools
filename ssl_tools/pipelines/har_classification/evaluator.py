#!/usr/bin/env python3

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
from typing import Dict
import lightning as L
import tqdm
import yaml
from ssl_tools.models.nets.simple import SimpleClassificationNet
from ssl_tools.pipelines.base import Pipeline
import torch
import torchmetrics
import pandas as pd
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import pickle

from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from io import BytesIO
from typing import List, Dict, final
import fsspec
import lightning as L
from lightning.pytorch.loggers import Logger, MLFlowLogger

from ssl_tools.callbacks.performance import PerformanceLogger

from lightning.pytorch.callbacks import (
    RichProgressBar,
)
from ssl_tools.pipelines.har_classification.utils import (
    FFT,
    Flatten,
    Spectrogram,
    DimensionAdder,
    SwapAxes,
)
from ssl_tools.pipelines.cli import auto_main
from ssl_tools.pipelines.base import Pipeline


from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.ssl.classifier import SSLDiscriminator


def get_split_dataloader(
    stage: str, data_module: L.LightningDataModule
) -> DataLoader:
    if stage == "train":
        data_module.setup("fit")
        return data_module.train_dataloader()
    elif stage == "validation":
        data_module.setup("fit")
        return data_module.val_dataloader()
    elif stage == "test":
        data_module.setup("test")
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Invalid stage: {stage}")


def full_dataset_from_dataloader(dataloader: DataLoader):
    return dataloader.dataset[:]


def get_full_data_split(
    data_module: L.LightningDataModule,
    stage: str,
):
    dataloader = get_split_dataloader(stage, data_module)
    return full_dataset_from_dataloader(dataloader)


def generate_embeddings(
    model: SimpleClassificationNet | SSLDiscriminator,
    dataloader: DataLoader,
    trainer: L.Trainer,
):
    if isinstance(model, SSLDiscriminator):
        old_head = model.head
        model.head = torch.nn.Identity()
    else:
        old_head = model.fc
        model.fc = torch.nn.Identity()

    embeddings = trainer.predict(model, dataloader)
    embeddings = torch.cat(embeddings)

    if isinstance(model, SSLDiscriminator):
        model.head = old_head
    else:
        model.fc = old_head

    return embeddings


import torch

import numpy as np
import plotly.graph_objects as go

import mlflow
from ssl_tools.pipelines.utils import load_model_mlflow
from mlflow.entities import ViewType
from tqdm.contrib.concurrent import process_map


class EmbeddingEvaluator(Pipeline):
    def __init__(
        self,
        # Required parameters
        experiment_name: str,
        registered_model_name: str,
        registered_model_tags: Dict[str, str] = None,
        experiment_tags: Dict[str, str] = None,
        n_classes: int = 7,
        # Optional parameters
        run_name: str = None,
        accelerator: str = "cpu",
        devices: int = 1,
        num_nodes: int = 1,
        num_workers: int = None,
        strategy: str = "auto",
        batch_size: int = 1,
        limit_predict_batches: int | float = 1.0,
        log_dir: str = "./mlruns",
        results_file: str = "results.csv",
        confusion_matrix_file: str = "confusion_matrix.csv",
        confusion_matrix_image_file: str = "confusion_matrix.png",
        tsne_plot_file: str = "tsne_embeddings.png",
        embedding_file: str = "embeddings.csv",
        predictions_file: str = "predictions.csv",
        add_epoch_info: bool = False,
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.experiment_tags = experiment_tags
        self.registered_model_name = registered_model_name
        self.registered_model_tags = registered_model_tags
        self.n_classes = n_classes
        self.run_name = run_name
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.strategy = strategy
        self.batch_size = batch_size
        self.limit_predict_batches = limit_predict_batches
        self.log_dir = log_dir
        self.model_version = dict()
        self._mlflow_client = None

        self.results_file = results_file
        self.confusion_matrix_file = confusion_matrix_file
        self.confusion_matrix_image_file = confusion_matrix_image_file
        self.tsne_plot_file = tsne_plot_file
        self.embedding_file = embedding_file
        self.predictions_file = predictions_file
        self.add_epoch_info = add_epoch_info

        self._sklearn_models = {
            "random_forest-100": partial(
                RandomForestClassifier, n_estimators=100, random_state=42
            ),
            "svm": partial(SVC, random_state=42),
            "knn-5": partial(KNeighborsClassifier, n_neighbors=5),
        }

    @property
    def client(self):
        if self._mlflow_client is None:
            self._mlflow_client = mlflow.client.MlflowClient(
                tracking_uri=self.log_dir
            )
        return self._mlflow_client

    def get_data_module(self) -> L.LightningDataModule:
        raise NotImplementedError

    def get_trainer(
        self, logger: Logger, callbacks: List[L.Callback]
    ) -> L.Trainer:
        return L.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            num_nodes=self.num_nodes,
            strategy=self.strategy,
            logger=logger,
            callbacks=callbacks,
            limit_predict_batches=self.limit_predict_batches,
        )

    def get_callbacks(self) -> List[L.Callback]:
        callbacks = []

        perfomance_logger = PerformanceLogger()
        callbacks.append(perfomance_logger)

        rich_progress_bar = RichProgressBar()
        callbacks.append(rich_progress_bar)

        return callbacks

    def get_logger(self) -> Logger:
        return MLFlowLogger(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            save_dir=self.log_dir,
            log_model=False,
            tags=self.experiment_tags,
        )

    def load_model(self) -> L.LightningModule:
        model, model_version = load_model_mlflow(
            self.client,
            self.registered_model_name,
            self.registered_model_tags,
        )
        self.model_version = model_version
        self._hparams.update({"model_version": model_version})
        return model

    def _compute_classification_metrics(
        self, y_hat_logits: torch.Tensor, y: torch.Tensor, n_classes: int
    ) -> pd.DataFrame:
        # print("-------------------------------------------")
        # print(y_hat_logits.shape, y.shape, n_classes)
        # print("-------------------------------------------")
        results = {
            "accuracy": [
                torchmetrics.functional.accuracy(
                    y_hat_logits, y, num_classes=n_classes, task="multiclass"
                ).item()
            ],
            "f1": [
                torchmetrics.functional.f1_score(
                    y_hat_logits, y, num_classes=n_classes, task="multiclass"
                ).item()
            ],
        }
        return pd.DataFrame(results)

    def _confusion_matrix(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        n_classes: int,
    ) -> pd.DataFrame:
        cm = torchmetrics.functional.confusion_matrix(
            y_hat, y, num_classes=n_classes, normalize="true", task="multiclass"
        )
        cm = pd.DataFrame(cm)
        return cm

    def _plot_confusion_matrix(
        self, cm: pd.DataFrame, classes: List[int]
    ) -> go.Figure:
        fig = go.Figure(data=go.Heatmap(z=cm, x=classes, y=classes))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="True",
            legend=dict(title="Classes"),
            showlegend=True,
        )
        return fig

    def _plot_tnse_embeddings(
        self,
        embeddings: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        n_components: int = 2,
    ) -> go.Figure:
        tsne = TSNE(n_components=n_components)
        embeddings_tsne = tsne.fit_transform(embeddings)

        # Colorize embeddings based on y
        colors = y

        # Create a list to store marker symbols
        markers = []

        # Iterate over y and y_hat to determine marker symbols
        for i in range(len(y)):
            if y[i] == y_hat[i]:
                markers.append("circle")
            else:
                markers.append("cross")

        # Create a scatter plot
        fig = go.Figure()
        markers = np.array(markers)

        # Add markers to the scatter plot
        unique_labels = torch.unique(y)
        for label in unique_labels:
            mask = (y == label).squeeze()
            fig.add_trace(
                go.Scatter(
                    x=embeddings_tsne[mask, 0],
                    y=embeddings_tsne[mask, 1],
                    mode="markers",
                    name=f"Class {label.item()}",
                    marker=dict(color=label.item(), symbol=markers[mask]),
                )
            )

        fig.update_layout(
            title="T-SNE Embeddings",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            legend=dict(title="Classes"),
        )
        return fig

    def predict(self, model, dataloader, trainer):
        y_hat = trainer.predict(model, dataloader)
        y_hat = torch.cat(y_hat)
        return y_hat

    def evaluate_model_performance(self, model, data_module, trainer):
        run_id = trainer.logger.run_id

        for stage in ["validation", "test"]:
            # ------------ Generate required data ------------
            # Get dataloader
            dataloader = get_split_dataloader(stage, data_module)
            # Get labels
            _, y = full_dataset_from_dataloader(dataloader)
            y = torch.LongTensor(y)
            # Predict
            y_hat_logits = self.predict(model, dataloader, trainer)
            n_classes = y_hat_logits.shape[1]
            y_hat = torch.argmax(y_hat_logits, dim=1)
            # Get embeddings
            embeddings = generate_embeddings(model, dataloader, trainer)
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            # Get number of classes
            # n_classes = len(torch.unique(y))
            # n_classes = self.n_classes
            classes = list(range(n_classes))

            # ------------ Evaluation ------------
            with TemporaryDirectory(prefix="test") as temp_dir:
                temp_dir = Path(temp_dir)
                # Generate predictions CSV
                predictions = pd.DataFrame(
                    {
                        "y": y.numpy().reshape(-1),
                        "y_hat": y_hat.numpy().reshape(-1),
                    }
                )
                predictions.to_csv(
                    temp_dir / self.predictions_file, index=False
                )

                # Classification metrics
                classification_results = self._compute_classification_metrics(
                    y_hat_logits, y, n_classes
                )
                classification_results.to_csv(
                    temp_dir / self.results_file, index=False
                )
                # Iterate over classification results and log metrics
                for metric_name, metric_value in classification_results.items():
                    self.client.log_metric(
                        run_id, f"{stage}_{metric_name}", metric_value[0]
                    )

                # Confusion matrix
                cm = self._confusion_matrix(y_hat, y, n_classes)
                cm.to_csv(temp_dir / self.confusion_matrix_file, index=False)

                fig = self._plot_confusion_matrix(cm, classes)
                fig.write_image(
                    temp_dir / self.confusion_matrix_image_file,
                    width=1.5 * 600,
                    height=1.5 * 600,
                    scale=1,
                )

                fig = self._plot_tnse_embeddings(
                    embeddings,
                    y,
                    y_hat,
                    n_components=2,
                )
                fig.write_image(
                    temp_dir / self.tsne_plot_file,
                    width=1.5 * 600,
                    height=1.5 * 600,
                    scale=1,
                )

                artifact_path = f"{stage}_set"
                if self.add_epoch_info:
                    artifact_path = (
                        f"epoch_{trainer.current_epoch}/{artifact_path}"
                    )

                # Log artifacts
                self.client.log_artifacts(
                    run_id=run_id,
                    local_dir=temp_dir,
                    artifact_path=artifact_path,
                )

    def _evaluate_embeddings(
        self,
        model,
        y_hat,
        y,
        n_classes,
        run_id,
        artifact_path,
    ):
        classes = list(range(n_classes))
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            # pickle_path = temp_dir / "model.pkl"
            # with open(pickle_path, "wb") as f:
            #     pickle.dump(model, f)

            # Generate predictions CSV
            predictions = pd.DataFrame(
                {
                    "y": y.reshape(-1),
                    "y_hat": y_hat.reshape(-1),
                }
            )
            predictions.to_csv(temp_dir / self.predictions_file, index=False)

            # Classification metrics
            classification_results = self._compute_classification_metrics(
                torch.LongTensor(y_hat),
                torch.LongTensor(y),
                n_classes,
            )
            classification_results.to_csv(
                temp_dir / self.results_file, index=False
            )

            # Confusion matrix
            cm = self._confusion_matrix(
                torch.LongTensor(y_hat), torch.LongTensor(y), n_classes
            )
            cm.to_csv(temp_dir / self.confusion_matrix_file, index=False)

            fig = self._plot_confusion_matrix(cm, classes)
            fig.write_image(
                temp_dir / self.confusion_matrix_image_file,
                width=1.5 * 600,
                height=1.5 * 600,
                scale=1,
            )

            # print(f"Using artifact path: {artifact_path}" )

            # Log artifacts
            self.client.log_artifacts(
                run_id=run_id,
                local_dir=temp_dir,
                artifact_path=artifact_path,
            )

    def evaluate_embeddings(self, model, data_module, trainer):
        run_id = trainer.logger.run_id
        train_loader = get_split_dataloader("train", data_module)
        val_loader = get_split_dataloader("validation", data_module)
        test_loader = get_split_dataloader("test", data_module)

        # Generate embeddings
        X_train_emb = generate_embeddings(model, train_loader, trainer)
        X_train_emb = X_train_emb.reshape(X_train_emb.shape[0], -1)

        X_val_emb = generate_embeddings(model, val_loader, trainer)
        X_val_emb = X_val_emb.reshape(X_val_emb.shape[0], -1)

        X_test_emb = generate_embeddings(model, test_loader, trainer)
        X_test_emb = X_test_emb.reshape(X_test_emb.shape[0], -1)

        # Get data
        X_train, y_train = full_dataset_from_dataloader(train_loader)
        X_val, y_val = full_dataset_from_dataloader(val_loader)
        X_test, y_test = full_dataset_from_dataloader(test_loader)

        # Get number of classes
        # n_classes = len(np.unique(y_train))
        n_classes = self.n_classes

        # Train using sklearn models
        for model_name, model_cls in self._sklearn_models.items():
            ### Train on train, test on validation
            model = model_cls()
            model.fit(X_train_emb, y_train)
            y_hat_val = model.predict(X_val_emb)
            artifact_path = f"sklearn_{model_name}/train/validation"
            if self.add_epoch_info:
                artifact_path = f"epoch_{trainer.current_epoch}/{artifact_path}"

            self._evaluate_embeddings(
                model=model,
                y_hat=y_hat_val,
                y=y_val,
                n_classes=n_classes,
                run_id=run_id,
                artifact_path=artifact_path,
            )

            ### Train on train, test on test
            artifact_path = f"sklearn_{model_name}/train/test"
            if self.add_epoch_info:
                artifact_path = f"epoch_{trainer.current_epoch}/{artifact_path}"
            y_hat_test = model.predict(X_test_emb)
            self._evaluate_embeddings(
                model=model,
                y_hat=y_hat_test,
                y=y_test,
                n_classes=n_classes,
                run_id=run_id,
                artifact_path=artifact_path,
            )

        # Concatenate train and validation
        X_train_val_emb = torch.cat([X_train_emb, X_val_emb])
        y_train_val = np.concatenate([y_train, y_val])

        for model_name, model_cls in self._sklearn_models.items():
            model = model_cls()
            model.fit(X_train_val_emb, y_train_val)
            y_hat_test = model.predict(X_test_emb)

            ### Train on train, test on test
            artifact_path = f"sklearn_{model_name}/train+val/test"
            if self.add_epoch_info:
                artifact_path = f"epoch_{trainer.current_epoch}/{artifact_path}"

            y_hat_test = model.predict(X_test_emb)
            self._evaluate_embeddings(
                model=model,
                y_hat=y_hat_test,
                y=y_test,
                n_classes=n_classes,
                run_id=run_id,
                artifact_path=artifact_path,
            )

    def run_task(
        self,
        model: SimpleClassificationNet | SSLDiscriminator,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ):
        self.evaluate_model_performance(model, data_module, trainer)
        # self.evaluate_embeddings(model, data_module, trainer)

    def run(self):
        # Get all required components
        model = self.load_model()
        datamodule = self.get_data_module()
        logger = self.get_logger()
        callbacks = self.get_callbacks()
        trainer = self.get_trainer(logger, callbacks)

        # Log the experiment hyperparameters
        hparams = dict(self.hparams)
        hparams.update({"model_version": self.model_version})
        logger.log_hyperparams(hparams)

        # Run task
        return self.run_task(model, datamodule, trainer)


transforms_map = {
    "identity": lambda x: x,
    "flatten": Flatten(),
    "fft": FFT(absolute=True, centered=True),
    "spectrogram": Spectrogram(),
    "dimension_adder": DimensionAdder(dim=1),
    "swapaxes": SwapAxes(0, 1),
}


class HAREmbeddingEvaluator(EmbeddingEvaluator):
    def __init__(
        self, data: str, transforms: str | List[str] = "identity", **kwargs
    ):
        super().__init__(**kwargs)
        self.data = data
        self.transforms = None
        if isinstance(transforms, str):
            self.transforms = [transforms_map[transforms]]
        else:
            self.transforms = [
                transforms_map[transform] for transform in transforms
            ]

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


def run_evaluator_wrapper(evaluator: EmbeddingEvaluator):
    # return evaluator.run()
    try:
        return evaluator.run()
    except Exception as e:
        print(f" ------- Error running evaluator: {e} ----------")
        traceback.print_exc()
        print("----------------------------------------------------")
        raise e


class EvaluateAll(Pipeline):
    def __init__(
        self,
        root_dataset_dir: str,
        experiment_id: str | List[str],  # The experiment(s) to evaluate
        experiment_names: str | List[str],  # Name of the experiment (result)
        config_dir: str = None,
        log_dir: str = "./mlruns",
        skip_existing: bool = True,
        accelerator: str = "cpu",
        devices: int = 1,
        num_nodes: int = 1,
        num_workers: int = None,
        strategy: str = "auto",
        batch_size: int = 1,
        use_ray: bool = False,
        ray_address: str = None,
    ):
        self.root_dataset_dir = Path(root_dataset_dir)
        self.config_dir = Path(config_dir) if config_dir is not None else None
        self.experiment_id = experiment_id
        self.experiment_names = experiment_names
        if isinstance(experiment_names, list):
            self.experiment_name = experiment_names[0]
        else:
            self.experiment_name = experiment_names
        self.skip_existing = skip_existing
        self.log_dir = log_dir
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count()
        )
        self.strategy = strategy
        self.batch_size = batch_size
        self.use_ray = use_ray
        self.ray_address = ray_address

        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = mlflow.client.MlflowClient(tracking_uri=self.log_dir)
        return self._client

    def summarize(self, runs):
        res = []
        for r in runs:
            if r.info.status != "FINISHED":
                continue
            d = dict(
                # Run info
                run_id=r.info.run_id,
                # Tags
                model=r.data.tags["model"],
                trained_on=r.data.tags["trained_on"],
                stage=r.data.tags["stage"],
                finetune_on=r.data.tags.get("finetune_on", ""),
                test_on=r.data.tags.get("test_on", ""),
                update_backbone=r.data.params.get("update_backbone", "False"),
            )

            d["update_backbone"] = d["update_backbone"] == "True"
            d["freeze"] = not d["update_backbone"]
            res.append(d)
        return pd.DataFrame(res)

    def locate_config(self, model_name):
        if self.config_dir is None:
            return dict()
        config_file = self.config_dir / f"{model_name}.yaml"
        if not config_file.exists():
            return dict()
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            # print(f"Loaded configuration from: {config_file}")
            return config

    def get_runs(self, experiment_ids, search_string: str = "") -> pd.DataFrame:
        runs = self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=search_string,
            max_results=50000,
            order_by=["start_time DESC"],
        )
        runs = self.summarize(runs)
        return runs

    def filter_runs(self, runs):
        experiment_ids = [
            self.client.get_experiment_by_name(experiment).experiment_id
            for experiment in self.experiment_names
        ]
        already_executed_runs = self.get_runs(
            experiment_ids,
            search_string="tags.`stage` = 'test'",
        )
        runs = runs[
            ~runs[["model", "trained_on", "finetune_on", "update_backbone"]]
            .apply(tuple, axis=1)
            .isin(
                already_executed_runs[
                    ["model", "trained_on", "finetune_on", "update_backbone"]
                ].apply(tuple, axis=1)
            )
        ]
        return runs

    def run(self):
        runs = self.get_runs(
            self.experiment_id, search_string="tags.`stage` != 'test'"
        )

        if self.skip_existing:
            runs = self.filter_runs(runs)

        configs_to_run = []

        for dataset_dir in tqdm.tqdm(self.root_dataset_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for i, (row_idx, row) in enumerate(runs.iterrows()):
                model = row["model"]
                trained_on = row["trained_on"]

                evaluator_kwargs = {
                    "experiment_name": self.experiment_name,
                    "registered_model_name": model,
                    "registered_model_tags": {
                        "model": model,
                        "trained_on": trained_on,
                        "stage": row["stage"],
                        # "finetune_on": finetune_on,
                    },
                    "experiment_tags": {
                        "model": model,
                        "trained_on": trained_on,
                        # "finetune_on": finetune_on,
                        "test_on": str(dataset_dir.name),
                        "stage": "test",
                        # "freeze": not update_backbone,
                        # "update_backbone": str(update_backbone),
                    },
                    "log_dir": self.log_dir,
                    "data": str(dataset_dir),
                    "num_workers": self.num_workers,
                    "accelerator": self.accelerator,
                    "devices": self.devices,
                    "num_nodes": self.num_nodes,
                    "strategy": self.strategy,
                    "batch_size": self.batch_size,
                }

                if row["finetune_on"] != "":
                    evaluator_kwargs["registered_model_tags"]["finetune_on"] = (
                        row["finetune_on"]
                    )
                    evaluator_kwargs["experiment_tags"]["finetune_on"] = row[
                        "finetune_on"
                    ]
                    evaluator_kwargs["experiment_tags"]["update_backbone"] = (
                        str(row["update_backbone"])
                    )

                config = self.locate_config(model)
                evaluator_kwargs["transforms"] = config.get(
                    "transforms", "identity"
                )
                evaluator = HAREmbeddingEvaluator(**evaluator_kwargs)
                configs_to_run.append(evaluator)
                
        print(f"Found {len(configs_to_run)} configurations to run")

        if self.use_ray:
            import ray

            ray.init(address=self.ray_address)
            remotes_to_run = [
                ray.remote(
                    num_gpus=0.10,
                    num_cpus=2,
                    max_calls=1,
                    max_retries=0,
                    retry_exceptions=False,
                )(run_evaluator_wrapper).remote(evaluator)
                for evaluator in configs_to_run
            ]
            ready, not_ready = ray.wait(
                remotes_to_run, num_returns=len(remotes_to_run)
            )
            print(f"Ready: {len(ready)}. Not ready: {len(not_ready)}")
            ray.shutdown()
        else:
            for i, evaluator in enumerate(configs_to_run):
                print(f"Running evaluator {i+1}/{len(configs_to_run)}")
                evaluator.run()
                print("----------------------------------------------------")
                
                
            results = process_map(
                run_evaluator_wrapper, configs_to_run, max_workers=4
            )
            ok = sum(r * 1 for r in results)
            not_ok = len(results) - ok
            print(f"OK: {ok}/{len(results)}. Not OK: {not_ok}/{len(results)}")


class CSVGenerator(Pipeline):
    def __init__(
        self,
        experiments: str | List[str],
        log_dir: str = "./mlruns",
        results_file: str = "results.csv",
    ):
        self.experiments = experiments
        self.log_dir = log_dir
        self.results_file = Path(results_file)
        self._mlflow_client: mlflow.client.MlflowClient = None

    @property
    def client(self):
        if self._mlflow_client is None:
            self._mlflow_client = mlflow.client.MlflowClient(
                tracking_uri=self.log_dir
            )
        return self._mlflow_client

    def run(self):
        runs = self.client.search_runs(
            experiment_ids=self.experiments,
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=50000,
            order_by=["tags.model ASC"],
        )
        if len(runs) == 0:
            raise ValueError("No runs found")

        results = []
        for run in runs:
            if run.info.status != "FINISHED":
                continue
            if "test_accuracy" not in run.data.metrics:
                continue
            
            # print(run)
            
            d = {
                "model": run.data.tags["model"],
                "test_accuracy": run.data.metrics["test_accuracy"],
                "validation_accuracy": run.data.metrics["validation_accuracy"],
                "update_backbone": run.data.tags.get("update_backbone", ""),
                "trained_on": run.data.tags["trained_on"],
                "finetune_on": run.data.tags.get("finetune_on", ""),
                "test_on": run.data.tags["test_on"],
                "run_name": run.info.run_name,
            }

            results.append(d)

        df = pd.DataFrame(results)
        test_datasets = sorted(df["test_on"].unique())
        print(f"There are {len(df)} results")

        results = []
        for (model, train, finetune, update_backbone), group in df.groupby(
            ["model", "trained_on", "finetune_on", "update_backbone"]
        ):
            freeze = ""
            # print(f"Update backbone: {update_backbone}")
            if finetune == "":
                freeze = ""
            else:
                if update_backbone == "":
                    freeze = True
                else:
                    if update_backbone.lower() == "true":
                        freeze = False
                    else:
                        freeze = True

            # test_acc_mean = group["test_accuracy"].mean()
            # val_acc_mean = group["validation_accuracy"].mean()
            row = [model, train, finetune, freeze]
            for test_set in test_datasets:
                test_acc = group[group["test_on"] == test_set]["test_accuracy"]
                if len(test_acc) == 0:
                    test_acc = np.nan
                else:
                    test_acc = test_acc.mean()
                row.append(test_acc)

            # row.append(test_acc_mean)
            results.append(row)

        columns = [
            "model",
            "trained_on",
            "finetune_on",
            "freeze",
        ] + test_datasets
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(self.results_file, index=False)
        print(f"Results saved to {self.results_file.resolve()}")


if __name__ == "__main__":
    options = {
        "evaluate": HAREmbeddingEvaluator,
        "evaluate-all": EvaluateAll,
        "csv": CSVGenerator,
    }

    auto_main(options)
