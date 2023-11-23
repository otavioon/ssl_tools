#!/usr/bin/env python

# TODO: A way of removing the need to add the path to the root of
# the project
from pathlib import Path
import sys
import os
import numpy as np

import pandas as pd

sys.path.append("../../../")


from typing import Union
import torch
from datetime import datetime
from jsonargparse import CLI

from ssl_tools.networks.layers.gru import GRUEncoder
from ssl_tools.utils.lightining_logger import performance_lightining_logger
from pytorch_lightning.loggers import CSVLogger
from ssl_tools.data.simple import SimpleDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


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
        """
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.experiment_name = name
        self.experiment_version = version

    def _train_lightning_module(self, model):
        pass

    def _train_torch(self, model):
        pass

    def _get_data(self):
        data_path = Path(self.data)
        x_train = []
        for f in data_path.glob("*.csv"):
            data = pd.read_csv(f)
            x = data[
                ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
            ].values
            x = np.swapaxes(x, 1, 0)
            x_train.append(x)
        dataset = SimpleDataset(x_train)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

        return dataloader

    def _get_logger(self):
        logger = CSVLogger(
            save_dir=self.log_dir,
            name=self.experiment_name,
            version=self.experiment_version,
            flush_logs_every_n_steps=100,
        )
        return logger

    def cpc(self, encoding_size: int = 10, window_size: int = 4):
        from ssl_tools.ssl.system.cpc import CPC

        # Set the experiment name and version
        self.experiment_name = self.experiment_name or "CPC"
        self.experiment_version = self.experiment_version or datetime.now().strftime(
            "%Y%m%d.%H%M%S"
        )

        # build the model
        encoder = GRUEncoder(encoding_size=encoding_size)
        density_estimator = torch.nn.Linear(encoding_size, encoding_size)
        auto_regressor = torch.nn.GRU(
            input_size=encoding_size, hidden_size=encoding_size, batch_first=True
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
        dataloader = self._get_data()
        # Get the logger
        logger = self._get_logger()

        # Get the hyperparameters and log them
        hyperparams = self.__dict__.copy()
        hyperparams.update(model.get_config())
        logger.log_hyperparams(hyperparams)

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=logger,
            enable_checkpointing=True,
            # TODO do this automatically
            accelerator="gpu",
            devices=1,
            strategy="ddp",
            # profiler="simple",
        )

        # Start training
        print(
            f"** Start training model CPC for {self.epochs} epochs. The name of the experiment name is {self.experiment_name} and version is {self.experiment_version} **"
        )
        trainer.fit(model, dataloader)

    def tnc(
        self, encoding_size: int = 10, window_size: int = 4, mc_sample_size: int = 20
    ):
        pass

    def tfc(
        self,
        length_alignment: int = 178,
        use_cosine_similarity: bool = True,
        temperature: float = 0.5,
    ):
        pass


if __name__ == "__main__":
    CLI(PretrainerMain, as_positional=False)
