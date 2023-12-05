import torch

from datetime import datetime
from jsonargparse import CLI

from ssl_tools.data.data_modules import (
    MultiModalHARDataModule,
    TNCHARDataModule,
    TFCDataModule,
    HARDataModule,
)
from ssl_tools.apps import LightningTrainCLI
from ssl_tools.callbacks.performance import PerformanceLog
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from torchmetrics import Accuracy

from .lightning_cli import LightningTrainCLI

from pathlib import Path


class SSLTrain(LightningTrainCLI):
    _MODEL_NAME = "model"
    _EXPERIMENT_VERSION_FORMAT = "%Y-%m-%d_%H-%M-%S"

    def __init__(
        self,
        training_mode: str = "pretrain",
        load_backbone: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.training_mode = training_mode
        self.load_backbone = load_backbone

        self.experiment_path = None
        self.checkpoint_path = None

    def _set_experiment(self):
        self.log_dir = Path(self.log_dir)

        if self.experiment_name is None:
            self.experiment_name = self._MODEL_NAME
        if self.experiment_version is None:
            self.experiment_version = datetime.now().strftime(
                self._EXPERIMENT_VERSION_FORMAT
            )

        self.experiment_path = (
            self.log_dir
            / self.experiment_name
            / self.training_mode
            / self.experiment_version
        )
        self.checkpoint_path = self.experiment_path / "checkpoints"

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
            monitor=self.checkpoint_metric,
            mode=self.checkpoint_metric_mode,
            dirpath=self.checkpoint_path,
            save_last=True,
        )

        performance_log = PerformanceLog()

        rich_progress_bar = RichProgressBar(
            leave=False, console_kwargs={"soft_wrap": True}
        )

        return [checkpoint_callback, rich_progress_bar, performance_log]

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

    def _load_model(self, model: L.LightningModule, path: str):
        print(f"Loading model from: {path}")
        state_dict = torch.load(path)["state_dict"]
        model.load_state_dict(state_dict)
        print("Model loaded successfully")

    def _log_hyperparams(self, model, logger):
        hyperparams = self.__dict__.copy()
        if getattr(model, "get_config", None):
            hyperparams.update(model.get_config())
        logger.log_hyperparams(hyperparams)
        return hyperparams

    def _get_pretrain_model(self) -> L.LightningModule:
        raise NotImplementedError

    def _get_pretrain_data_module(self) -> L.LightningDataModule:
        raise NotImplementedError

    def _get_finetune_model(
        self, load_backbone: str = None
    ) -> L.LightningModule:
        raise NotImplementedError

    def _get_finetune_data_module(self) -> L.LightningDataModule:
        raise NotImplementedError

    def _train(
        self,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ):        
        return trainer.fit(model, data_module, ckpt_path=self.resume)

    def _run(self):
        # ----------------------------------------------------------------------
        # 1. Set experiment name and version
        # ----------------------------------------------------------------------
        self._set_experiment()

        # ----------------------------------------------------------------------
        # 2. Instantiate model and data module
        # ----------------------------------------------------------------------
        if self.training_mode == "pretrain":
            model = self._get_pretrain_model()
            data_module = self._get_pretrain_data_module()
        else:
            model = self._get_finetune_model(load_backbone=self.load_backbone)
            data_module = self._get_finetune_data_module()
            
        if self.load is not None:
            self._load_model(model, self.load)

        # ----------------------------------------------------------------------
        # 3. Instantiate trainer specific resources (logger, callbacks, etc.)
        # ----------------------------------------------------------------------
        logger = self._get_logger()
        callbacks = self._get_callbacks()

        # ----------------------------------------------------------------------
        # 4. Log the hyperparameters (for reproducibility purposes)
        # ----------------------------------------------------------------------
        hyperparams = self._log_hyperparams(model, logger)

        # ----------------------------------------------------------------------
        # 5. Instantiate the trainer
        # ----------------------------------------------------------------------
        trainer = self._get_trainer(logger, callbacks)

        # ----------------------------------------------------------------------
        # 6. Train the model
        # ----------------------------------------------------------------------
        self._train(model, data_module, trainer)

    def __call__(self):
        return self._run()
