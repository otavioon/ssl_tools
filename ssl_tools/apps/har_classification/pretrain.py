#!/usr/bin/env python

# TODO: A way of removing the need to add the path to the root of
# the project
import sys


sys.path.append("../../../")


from typing import Union
import torch
from datetime import datetime
from jsonargparse import CLI

from ssl_tools.models.layers import GRUEncoder
from ssl_tools.utils.lightining_logger import performance_lightining_logger
from ssl_tools.data.data_modules import (
    MultiModalHARDataModule,
    TNCHARDataModule,
)
from pytorch_lightning.loggers import CSVLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from ssl_tools.apps import LightningTrainCLI
from ssl_tools.models.layers.linear import Discriminator


class LightningPretrainCLI(LightningTrainCLI):    
    """Defines a Main CLI for pre-training Self-supervised Pytorch Lightning 
    models

    Model and its speficic parameters are defined inner functions. Any inner 
    function is callable from the command line and its parameters are exposed 
    as command line arguments. Functions with names beginning with an underscore
    are not callable from the command line.
    
    In general, the train of models is done as follows:
        1. Assert the validity of the parameters
        2. Set the experiment name and version
        3. Instantiate the model
        4. Instantiate the data modules
        5. Instantiate trainer specific resources (logger, callbacks, etc.)
        6. Log the hyperparameters (for reproducibility purposes)
        7. Instantiate the trainer
        8. Train the model
    """   
    def _set_experiment(self, model_name: str):
        self.experiment_name = self.experiment_name or model_name
        self.experiment_version = (
            self.experiment_version or datetime.now().strftime("%Y%m%d.%H%M%S")
        )
    
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
    
    def _log_hyperparams(self, model, logger):
        hyperparams = self.__dict__.copy()
        if getattr(model, "get_config", None):
            hyperparams.update(model.get_config())
        logger.log_hyperparams(hyperparams)
        return hyperparams
    
    def _get_trainer(self, logger, callbacks):
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
        return trainer
    
    def _train(self, model, data_module, trainer):
        print(
            f"** Start training. \n" + \
            f"\tExperiment: {self.experiment_name} \n" + \
            f"\tVersion is {self.experiment_version} **"
        )
        
        return trainer.fit(model, data_module)

    def cpc(
        self,
        encoding_size: int = 10,
        window_size: int = 4,
        pad_length: bool = False,
    ):
        from ssl_tools.models.ssl import CPC
        # Wraps CPC in a lightning module for logging purposes
        CPC = performance_lightining_logger(CPC)
        
        # ----------------------------------------------------------------------
        # 1. Assert the validity of the parameters
        # ----------------------------------------------------------------------
        assert self.batch_size == 1, (
            "CPC only supports batch size of 1. Please set batch_size=1"
        )
        
        # ----------------------------------------------------------------------
        # 2. Set experiment name and version
        # ----------------------------------------------------------------------
        self._set_experiment("CPC_Pretrain")

        # ----------------------------------------------------------------------
        # 3. Instantiate model
        # ----------------------------------------------------------------------
        encoder = GRUEncoder(encoding_size=encoding_size)
        density_estimator = torch.nn.Linear(encoding_size, encoding_size)
        auto_regressor = torch.nn.GRU(
            input_size=encoding_size,
            hidden_size=encoding_size,
            batch_first=True,
        )
        model = CPC(
            encoder=encoder,
            density_estimator=density_estimator,
            auto_regressor=auto_regressor,
            window_size=window_size,
            lr=self.learning_rate,
        )

        # ----------------------------------------------------------------------
        # 4. Instantiate data modules
        # ----------------------------------------------------------------------
        data_module = MultiModalHARDataModule(
            self.data, batch_size=self.batch_size, fix_length=pad_length
        )

        # ----------------------------------------------------------------------
        # 5. Instantiate trainer specific resources (logger, callbacks, etc.)
        # ----------------------------------------------------------------------
        logger = self._get_logger()
        callbacks = self._get_callbacks()
        
        # ----------------------------------------------------------------------
        # 6. Log the hyperparameters (for reproducibility purposes)
        # ----------------------------------------------------------------------
        hyperparams = self._log_hyperparams(model, logger)
        
        # ----------------------------------------------------------------------
        # 7. Instantiate the trainer
        # ----------------------------------------------------------------------
        trainer = self._get_trainer(logger, callbacks)
        
        # ----------------------------------------------------------------------
        # 8. Train the model
        # ----------------------------------------------------------------------
        self._train(model, data_module, trainer)

    def tnc(
        self,
        encoding_size: int = 10,
        window_size: int = 60,
        mc_sample_size: int = 20,
        significance_level: float = 0.01,
        repeat: int = 5,
        pad_length: bool = True,
    ):
        from ssl_tools.models.ssl import TNC
        # Wraps TNC in a lightning module for logging purposes
        TNC = performance_lightining_logger(TNC)
        
        # ----------------------------------------------------------------------
        # 1. Assert the validity of the parameters
        # ----------------------------------------------------------------------
        assert significance_level > 0 and significance_level < 1, (
            "The significance level must be between 0 and 1"
        )
        
        # ----------------------------------------------------------------------
        # 2. Set experiment name and version
        # ----------------------------------------------------------------------
        self._set_experiment("TNC_Pretrain")

        # ----------------------------------------------------------------------
        # 3. Instantiate model
        # ----------------------------------------------------------------------
        discriminator = Discriminator(input_size=encoding_size)
        encoder = GRUEncoder(encoding_size=encoding_size)
        model = TNC(
            discriminator=discriminator,
            encoder=encoder,
            mc_sample_size=mc_sample_size,
            w=0.05,
            learning_rate=self.learning_rate,
        )

        # ----------------------------------------------------------------------
        # 4. Instantiate data modules
        # ----------------------------------------------------------------------
        data_module = TNCHARDataModule(
            self.data,
            batch_size=self.batch_size,
            fix_length=pad_length,
            window_size=window_size,
            mc_sample_size=mc_sample_size,
            significance_level=significance_level,
            repeat=repeat,
        )

        # ----------------------------------------------------------------------
        # 5. Instantiate trainer specific resources (logger, callbacks, etc.)
        # ----------------------------------------------------------------------
        logger = self._get_logger()
        callbacks = self._get_callbacks()
        
        # ----------------------------------------------------------------------
        # 6. Log the hyperparameters (for reproducibility purposes)
        # ----------------------------------------------------------------------
        hyperparams = self._log_hyperparams(model, logger)

        # ----------------------------------------------------------------------
        # 7. Instantiate the trainer
        # ----------------------------------------------------------------------
        trainer = self._get_trainer(logger, callbacks)
        
        # ----------------------------------------------------------------------
        # 8. Train the model
        # ----------------------------------------------------------------------
        self._train(model, data_module, trainer)

    def tfc(
        self,
        length_alignment: int = 178,
        use_cosine_similarity: bool = True,
        temperature: float = 0.5,
    ):
        pass


if __name__ == "__main__":
    CLI(LightningPretrainCLI, as_positional=False)
