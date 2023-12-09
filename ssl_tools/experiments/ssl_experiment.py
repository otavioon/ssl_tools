from typing import List, Union
import torch

from datetime import datetime

from ssl_tools.experiments import LightningTrain
from ssl_tools.callbacks.performance import PerformanceLog
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar

from .lightning_cli import LightningTrain, LightningTest

from pathlib import Path
import pandas as pd


class SSLTrain(LightningTrain):
    _MODEL_NAME = "model"

    def __init__(
        self,
        training_mode: str = "pretrain",
        load_backbone: str = None,
        *args,
        **kwargs,
    ):
        """Wraps the LightningTrain class to provide a more specific interface
        for SSL experiments (training).

        Parameters
        ----------
        training_mode : str, optional
            The training mode. It could be either "pretrain" or "finetune"
        load_backbone : str, optional
            Path to the backbone to load. This is only used when training_mode
            is "finetune". In fine-tuning, the backbone is loaded and the
            using ``load_backbone``. The ``load`` parameter is used to load the
            full model (backbone + head).
        """
        super().__init__(*args, **kwargs)
        self.training_mode = training_mode
        self.load_backbone = load_backbone
        self.checkpoint_path = None
        self.experiment_path = None

        assert self.training_mode in ["pretrain", "finetune"]

    def _set_experiment(self):
        """Set the experiment name and version. This method is called before
        instantiating the model and data module. It is used to set the
        experiment path and checkpoint path. The experiment path is used to
        store the logs and checkpoints. The checkpoint path is used to store
        the checkpoints.
        """
        if self.seed is not None:
            L.seed_everything(self.seed)

        self.log_dir = Path(self.log_dir) / self.training_mode

        if self.experiment_name is None:
            self.experiment_name = self._MODEL_NAME

        # Same as format as logger
        self.experiment_path = (
            self.log_dir / self.experiment_name / self.experiment_version
        )

        self.checkpoint_path = self.experiment_path / "checkpoints"

    def _get_logger(self):
        """Get the logger to use for the experiment.

        Returns
        -------
        __type__
            The logger to use for the experiment
        """
        logger = CSVLogger(
            save_dir=self.log_dir,
            name=self.experiment_name,
            version=self.experiment_version,
            # flush_logs_every_n_steps=100,
        )
        return logger

    def _get_callbacks(self) -> List[L.Callback]:
        """Get the callbacks to use for the experiment.

        Returns
        -------
        List[L.Callback]
            A list of callbacks to use for the experiment
        """
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

    def _get_trainer(self, logger, callbacks: List[L.Callback]) -> L.Trainer:
        """Get trainer to use for the experiment.

        Parameters
        ----------
        logger : _type_
            The logger to use for the experiment
        callbacks : List[L.Callback]
            A list of callbacks to use for the experiment

        Returns
        -------
        L.Trainer
            The trainer to use for the experiment
        """

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
        """Load a model from a checkpoint.

        Parameters
        ----------
        model : L.LightningModule
            The model to load the checkpoint into
        path : str
            The path to the checkpoint
        """
        print(f"Loading model from: {path}")
        state_dict = torch.load(path)["state_dict"]
        model.load_state_dict(state_dict)
        print("Model loaded successfully")

    def _log_hyperparams(self, model: L.LightningModule, logger):
        """Log the hyperparameters for reproducibility purposes.

        Parameters
        ----------
        model : L.LightningModule
            The model to log the hyperparameters from
        logger : _type_
            The logger to use for logging the hyperparameters
        """

        def nested_convert(data):
            if isinstance(data, dict):
                return {
                    key: nested_convert(value) for key, value in data.items()
                }
            elif isinstance(data, Path):
                return str(data.expanduser())
            else:
                return data

        hyperparams = self.__dict__.copy()
        if getattr(model, "get_config", None):
            hyperparams.update(model.get_config())
        hyperparams = nested_convert(hyperparams)
        logger.log_hyperparams(hyperparams)
        return hyperparams

    def _get_pretrain_model(self) -> L.LightningModule:
        """Get the model to use for pre-training.

        Returns
        -------
        L.LightningModule
            The model to use for pre-training
        """
        raise NotImplementedError

    def _get_pretrain_data_module(self) -> L.LightningDataModule:
        """The data module to use for pre-training.

        Returns
        -------
        L.LightningDataModule
            The data module to use for pre-training
        """
        raise NotImplementedError

    def _get_finetune_model(
        self, load_backbone: str = None
    ) -> L.LightningModule:
        """Get the model to use for fine-tuning.

        Parameters
        ----------
        load_backbone : str, optional
            The path to the backbone to load. The backbone must be loaded 
            inside this method, if it is not None.

        Returns
        -------
        L.LightningModule
            The model to use for fine-tuning
        """
        raise NotImplementedError

    def _get_finetune_data_module(self) -> L.LightningDataModule:
        """The data module to use for fine-tuning.

        Returns
        -------
        L.LightningDataModule
            The data module to use for fine-tuning

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def _train(
        self,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ) -> None:
        """Train the model using the provided trainer.

        Parameters
        ----------
        model : L.LightningModule
            The model to train
        data_module : L.LightningDataModule
            The data module to use for training
        trainer : L.Trainer
            The trainer to use for training
        """
        print(f"Training will start")
        print(f"\tExperiment path: {self.experiment_path}")

        return trainer.fit(model, data_module)

    def _run(self):
        """Runs the experiment. This method is called when the experiment is
        called as a function. This method:
        1. Sets the experiment name and version
        2. Instantiates the model and data module (depending on the 
        ``training_mode``)
        3. Instantiates the trainer specific resources (logger, callbacks, etc.)
        4. Logs the hyperparameters (for reproducibility purposes)
        5. Instantiates the trainer
        6. Trains the model
        """
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

        print(f"Training completed successfully.")
        print(f"Last checkpoint saved to: {self.checkpoint_path}/last.ckpt")

    def __call__(self):
        self._run()


class SSLTest(LightningTest):
    _MODEL_NAME = "model"

    def __init__(self, *args, **kwargs):
        """Wraps the LightningTest class to provide a more specific interface
        for SSL experiments (testing).
        """
        super().__init__(*args, **kwargs)
        self.experiment_path = None

    def _set_experiment(self):
        """Set the experiment name and version. This method is called before
        instantiating the model and data module. It sets the experiment path
        and results path. The experiment path is used to store the logs and 
        the results path is used to store the results.
        """
        if self.seed is not None:
            L.seed_everything(self.seed)
        
        self.log_dir = Path(self.log_dir) / "test"
        if self.experiment_name is None:
            self.experiment_name = self._MODEL_NAME

        # Same as format as logger
        self.experiment_path = (
            self.log_dir / self.experiment_name / self.experiment_version
        )

        self.results_path = self.experiment_path / "results.csv"

    def _get_logger(self):
        """Get the logger to use for the experiment.

        Returns
        -------
        _type_
            Get the logger to use for the experiment
        """
        logger = CSVLogger(
            save_dir=self.log_dir,
            name=self.experiment_name,
            version=self.experiment_version,
            # flush_logs_every_n_steps=100,
        )
        return logger

    def _get_callbacks(self) -> List[L.Callback]:
        """Get the callbacks to use for the experiment.

        Returns
        -------
        List[L.Callback]
            The list of callbacks to use for the experiment.
        """
        performance_log = PerformanceLog()
        rich_progress_bar = RichProgressBar(
            leave=False, console_kwargs={"soft_wrap": True}
        )
        return [rich_progress_bar, performance_log]

    def _get_trainer(self, logger, callbacks):
        trainer = L.Trainer(
            logger=logger,
            callbacks=callbacks,
            accelerator=self.accelerator,
            devices=self.devices,
            num_nodes=self.num_nodes,
            limit_test_batches=self.limit_test_batches,
        )
        return trainer

    def _load_model(self, model: L.LightningModule, path: str):
        """Loads a model from a checkpoint.

        Parameters
        ----------
        model : L.LightningModule
            The model to load the checkpoint into
        path : str
            The path to the checkpoint
        """
        print(f"Loading model from: {path}")
        state_dict = torch.load(path)["state_dict"]
        model.load_state_dict(state_dict)
        print("Model loaded successfully")

    def _log_hyperparams(self, model, logger):
        def nested_convert(data):
            if isinstance(data, dict):
                return {
                    key: nested_convert(value) for key, value in data.items()
                }
            elif isinstance(data, Path):
                return str(data.expanduser())
            else:
                return data

        hyperparams = self.__dict__.copy()
        if getattr(model, "get_config", None):
            hyperparams.update(model.get_config())
        hyperparams = nested_convert(hyperparams)
        logger.log_hyperparams(hyperparams)
        return hyperparams

    def _get_test_model(self) -> L.LightningModule:
        """Get the model to use for testing.

        Returns
        -------
        L.LightningModule
            The model to use for testing
        """
        raise NotImplementedError

    def _get_test_data_module(self) -> L.LightningDataModule:
        """The data module to use for testing.

        Returns
        -------
        L.LightningDataModule
            The data module to use for testing
        """
        raise NotImplementedError

    def _test(
        self,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ):
        """Test the model using the provided trainer.

        Parameters
        ----------
        model : L.LightningModule
            The model to test
        data_module : L.LightningDataModule
            The data module to use for testing
        trainer : L.Trainer
            The trainer to use for testing

        Returns
        -------
        _type_
            A list of dictionary with the results
        """
        return trainer.test(model, data_module)

    def _run(self) -> List[dict]:
        """Runs the experiment. This method is called when the experiment is
        called as a function. This method:
        1. Sets the experiment name and version
        2. Instantiates the model and data module
        3. Instantiates the trainer specific resources (logger, callbacks, etc.)
        4. Logs the hyperparameters (for reproducibility purposes)
        5. Instantiates the trainer
        6. Tests the model
        
        Note
        ----
            The results are converted to a pandas DataFrame and saved to the
            ``results_path``. The results are also returned by this method (as 
            a list of dictionaries).
        
        Returns
        -------
        List[dict]
            A list of dictionary with the results
        """
        # ----------------------------------------------------------------------
        # 1. Set experiment name and version
        # ----------------------------------------------------------------------
        self._set_experiment()

        # ----------------------------------------------------------------------
        # 2. Instantiate model and data module
        # ----------------------------------------------------------------------
        model = self._get_test_model()
        data_module = self._get_test_data_module()
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
        result = self._test(model, data_module, trainer)

        pd.DataFrame(result).to_csv(self.results_path, index=False)
        print(f"Results saved to: {self.results_path}")
        return result

    def __call__(self):
        return self._run()
