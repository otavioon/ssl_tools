#!/usr/bin/env python

# TODO: A way of removing the need to add the path to the root of
# the project
import sys


sys.path.append("../../../")


from typing import Union
import torch
from datetime import datetime
from jsonargparse import CLI

from ssl_tools.networks.layers.gru import GRUEncoder
from ssl_tools.utils.lightining_logger import performance_lightining_logger
from ssl_tools.data.data_modules.har_multi_csv import (
    MultiModalHARDataModule,
    TNCHARDataModule,
)
from pytorch_lightning.loggers import CSVLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np


class PretrainerMain:
    def __init__(
        self,
        data: str,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        log_dir: str = "logs",
        name: str = None,
        version: Union[str, int] = None,
        monitor_metric: str = None,
        accelerator: str = "cpu",
        devices: int = 1,
        strategy: str = "auto",
        limit_train_batches: Union[float, int] = 1.0,
        limit_val_batches: Union[float, int] = 1.0,
        num_nodes: int = 1,
    ):
        """Pre-train self-supervised models with a HAR dataset

        Parameters
        ----------
        data : str
            The location of the data
        epochs : int, optional
            Number of epochs to pre-train the model
        batch_size : int, optional
            The batch size
        learning_rate : float, optional
            The learning rate of the optimizer
        log_dir : str, optional
            Path to the location where logs will be stored
        name: str, optional
            The name of the experiment (will be used as a prefix for the logs and checkpoints). If not provided, the name of the model will be used
        version: Union[int, str], optional
            The version of the experiment. If not is provided the current date and time will be used as the version
        monitor_metric: str, optional
            The metric to monitor for checkpointing. If not provided, the last model will be saved
        accelerator: str, optional
            The accelerator to use. Defaults to "cpu"
        devices: int, optional
            The number of devices to use. Defaults to 1
        strategy: str, optional
            The strategy to use. Defaults to "auto"
        limit_train_batches: Union[float, int], optional
            The number of batches to use for training. Defaults to 1.0 (use all batches)
        limit_val_batches: Union[float, int], optional
            The number of batches to use for validation. Defaults to 1.0 (use all batches)
        num_nodes: int, optional
            The number of nodes to use. Defaults to 1


        """
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.experiment_name = name
        self.experiment_version = version
        self.monitor_metric = monitor_metric
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.num_nodes = num_nodes

    def _get_logger(self):
        logger = CSVLogger(
            save_dir=self.log_dir,
            name=self.experiment_name,
            version=self.experiment_version,
            # flush_logs_every_n_steps=100,
        )
        return logger

    def _get_callbacks(self):
        # Get the checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor=self.monitor_metric,
            mode="min",
            dirpath=f"{self.log_dir}/{self.experiment_name}/{self.experiment_version}/checkpoints",
            save_last=True,
        )
        return [checkpoint_callback]

    def cpc(
        self,
        encoding_size: int = 10,
        window_size: int = 4,
        pad_length: bool = False,
    ):
        from ssl_tools.ssl.system.cpc import CPC

        if self.batch_size != 1:
            raise ValueError(
                "CPC only supports batch size of 1. Please set batch_size=1"
            )

        # Set the experiment name and version
        self.experiment_name = self.experiment_name or "CPC"
        self.experiment_version = (
            self.experiment_version or datetime.now().strftime("%Y%m%d.%H%M%S")
        )

        # build the model
        encoder = GRUEncoder(encoding_size=encoding_size)
        density_estimator = torch.nn.Linear(encoding_size, encoding_size)
        auto_regressor = torch.nn.GRU(
            input_size=encoding_size,
            hidden_size=encoding_size,
            batch_first=True,
        )
        # Wraps CPC in a lightning module
        CPC = performance_lightining_logger(CPC)
        model = CPC(
            encoder=encoder,
            density_estimator=density_estimator,
            auto_regressor=auto_regressor,
            window_size=window_size,
            lr=self.learning_rate,
        )

        # Get the data
        data_module = MultiModalHARDataModule(
            self.data, batch_size=self.batch_size, fix_length=pad_length
        )

        # Get the logger
        logger = self._get_logger()

        # Get the callbacks
        callbacks = self._get_callbacks()

        # Get the hyperparameters and log them
        hyperparams = self.__dict__.copy()
        hyperparams.update(model.get_config())
        logger.log_hyperparams(hyperparams)

        # Set the trainer
        trainer = L.Trainer(
            max_epochs=self.epochs,
            logger=logger,
            # enable_checkpointing=True,
            callbacks=callbacks,
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            num_nodes=self.num_nodes,
        )

        # Start training
        print(
            f"** Start training model CPC for {self.epochs} epochs. The name of the experiment name is {self.experiment_name} and version is {self.experiment_version} **"
        )

        print(type(model))
        trainer.fit(model, data_module)

    def tnc_light(
        self,
        encoding_size: int = 10,
        window_size: int = 60,
        mc_sample_size: int = 20,
        significance_level: float = 0.01,
        repeat: int = 1,
        pad_length: bool = True,
    ):
        from ssl_tools.ssl.system.tnc import TNC
        from ssl_tools.ssl.builders.common import Discriminator

        # if self.batch_size != 1:
        #     raise ValueError(
        #         "CPC only supports batch size of 1. Please set batch_size=1"
        #     )

        # Set the experiment name and version
        self.experiment_name = self.experiment_name or "TNC"
        self.experiment_version = (
            self.experiment_version or datetime.now().strftime("%Y%m%d.%H%M%S")
        )

        ### Instantiate model
        discriminator = Discriminator(input_size=encoding_size)
        encoder = GRUEncoder(encoding_size=encoding_size)
        model = TNC(
            discriminator=discriminator,
            encoder=encoder,
            mc_sample_size=mc_sample_size,
            w=0.05,
            learning_rate=self.learning_rate,
        )

        #####################

        # Get the data
        data_module = TNCHARDataModule(
            self.data,
            batch_size=self.batch_size,
            fix_length=pad_length,
            window_size=window_size,
            mc_sample_size=mc_sample_size,
            significance_level=significance_level,
            repeat=repeat,
        )

        # Get the logger
        logger = self._get_logger()

        # Get the callbacks
        callbacks = self._get_callbacks()
        # Get the hyperparameters and log them
        hyperparams = self.__dict__.copy()
        hyperparams.update(model.get_config())
        logger.log_hyperparams(hyperparams)

        # Set the trainer
        trainer = L.Trainer(
            max_epochs=self.epochs,
            logger=logger,
            # enable_checkpointing=True,
            callbacks=callbacks,
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            num_nodes=self.num_nodes,
        )

        # Start training
        print(
            f"** Start training model CPC for {self.epochs} epochs. The name of the experiment name is {self.experiment_name} and version is {self.experiment_version} **"
        )

        print(type(model))
        trainer.fit(model, data_module)

    def tfc(
        self,
        length_alignment: int = 178,
        use_cosine_similarity: bool = True,
        temperature: float = 0.5,
    ):
        pass


if __name__ == "__main__":
    CLI(PretrainerMain, as_positional=False)
