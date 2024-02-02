from pathlib import Path
from typing import Any, List, Union
from abc import abstractmethod
import lightning as L
from lightning.pytorch.loggers import Logger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
import torch
from ssl_tools.callbacks.performance import PerformanceLog
from ssl_tools.experiments.experiment import Experiment

class LightningExperiment(Experiment):
    _MODEL_NAME: str = "model"
    _STAGE_NAME: str = "stage"

    def __init__(
        self,
        name: str = None,
        stage_name: str = None,
        batch_size: int = 1,
        load: str = None,
        accelerator: str = "cpu",
        devices: int = 1,
        strategy: str = "auto",
        num_nodes: int = 1,
        num_workers: int = None,
        log_every_n_steps: int = 50,
        *args,
        **kwargs,
    ):
        name = name or self._MODEL_NAME
        super().__init__(name=name, *args, **kwargs)
        
        self.stage_name = stage_name or self._STAGE_NAME
        self.batch_size = batch_size
        self.load = load
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.log_every_n_steps = log_every_n_steps
        
        self._model = None
        self._logger = None
        self._callbacks = None
        self._data_module = None
        self._trainer = None
        self._result = None
        self._run_count = 0

    @property
    def experiment_dir(self) -> Path:
        return (
            Path(self.log_dir) / self.stage_name / self.name / str(self.run_id)
        )

    @property
    def checkpoint_dir(self) -> Path:
        return self.experiment_dir / "checkpoints"
    
    @property
    def model(self) -> L.LightningModule:
        if self._model is None:
            self._model = self.get_model()
        return self._model
    
    @property
    def data_module(self) -> L.LightningDataModule:
        if self._data_module is None:
            self._data_module = self.get_data_module()
        return self._data_module
    
    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = self.get_logger()
        return self._logger
    
    @property
    def callbacks(self) ->List[L.Callback]:
        if self._callbacks is None:
            self._callbacks = self.get_callbacks()
        return self._callbacks
    
    @property
    def hyperparameters(self) -> dict:
        def nested_convert(data):
            if isinstance(data, dict):
                return {
                    key: nested_convert(value) for key, value in data.items() if not key.startswith("_")
                }
            elif isinstance(data, Path):
                return str(data.expanduser())
            else:
                return data

        hyperparams = self.__dict__.copy()
        
        
        if getattr(self.model, "get_config", None):
            hyperparams.update(self.model.get_config())
        hyperparams = nested_convert(hyperparams)
        return hyperparams

    @property
    def trainer(self) -> L.Trainer:
        if self._trainer is None:
            self._trainer = self.get_trainer(self.logger, self.callbacks)
        return self._trainer

    @property
    def finished(self) -> bool:
        return self._run_count > 0

    def setup(self):
        if self.seed is not None:
            L.seed_everything(self.seed)

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_logger(self) -> Logger:
        """Get the logger to use for the experiment.

        Returns
        -------
        Logger
            The logger to use for the experiment
        """
        experiment_dir = self.experiment_dir

        logger = CSVLogger(
            save_dir=experiment_dir.parents[1],
            name=self.experiment_dir.parents[0].name,
            version=self.experiment_dir.name,
        )
        return logger

    def get_callbacks(self) -> List[L.Callback]:
        """Get the callbacks to use for the experiment.

        Returns
        -------
        List[L.Callback]
            A list of callbacks to use for the experiment
        """
        return []

    def load_checkpoint(
        self, model: L.LightningModule, path: Path
    ) -> L.LightningModule:
        """Load the model to use for the experiment.

        Returns
        -------
        L.LightningModule
            The model to use for the experiment
        """
        print(f"Loading model from: {path}...")
        state_dict = torch.load(path)["state_dict"]
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
        return model

    def log_hyperparams(self, logger: Logger) -> dict:
        """Log the hyperparameters for reproducibility purposes.

        Parameters
        ----------
        model : L.LightningModule
            The model to log the hyperparameters from
        logger : Logger
            The logger to use for logging the hyperparameters
        """
        hparams = self.hyperparameters
        logger.log_hyperparams(hparams)

    def run(self):
        """Runs the experiment. This method:
        1. Instantiates the model and data module (depending on the
        ``training_mode``) and load the checkpoint if provided
        2. Instantiates the trainer specific resources (logger, callbacks, etc.)
        3. Logs the hyperparameters (for reproducibility purposes)
        4. Instantiates the trainer
        5. Trains/Tests the model
        """

        # ----------------------------------------------------------------------
        # 1. Instantiate model and data module
        # ----------------------------------------------------------------------
        model = self.model        
        data_module = self.data_module

        if self.load:
            model = self.load_checkpoint(model, self.load)

        # ----------------------------------------------------------------------
        # 2. Instantiate trainer specific resources (logger, callbacks, etc.)
        # ----------------------------------------------------------------------
        logger = self.logger

        # ----------------------------------------------------------------------
        # 3. Log the hyperparameters (for reproducibility purposes)
        # ----------------------------------------------------------------------
        self.log_hyperparams(logger)

        # ----------------------------------------------------------------------
        # 4. Instantiate the trainer
        # ----------------------------------------------------------------------
        trainer = self.trainer
        # ----------------------------------------------------------------------
        # 5. Train/Tests the model
        # ----------------------------------------------------------------------
        self._result = self.run_model(model, data_module, trainer)
        
        self._run_count += 1
        return self._result

    @abstractmethod
    def get_trainer(
        self, logger: Logger, callbacks: List[L.Callback]
    ) -> L.Trainer:
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
        raise NotImplementedError

    @abstractmethod
    def run_model(
        self,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> L.LightningModule:
        """Get the model to use for the experiment.

        Returns
        -------
        L.LightningModule
            The model to use for the experiment
        """
        raise NotImplementedError

    @abstractmethod
    def get_data_module(self) -> L.LightningDataModule:
        """Get the datamodule to use for the experiment.

        Returns
        -------
        L.LightningDataModule
            The datamodule to use for the experiment
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"LightningExperiment(experiment_dir={self.experiment_dir}, model={self._MODEL_NAME}, run_id={self.run_id}, finished={self.finished})"


class LightningTrain(LightningExperiment):
    _STAGE_NAME="train"
    
    def __init__(
        self,
        stage_name: str = "train",
        epochs: int = 1,
        learning_rate: float = 1e-3,
        checkpoint_metric: str = None,
        checkpoint_metric_mode: str = "min",
        limit_train_batches: Union[float, int] = 1.0,
        limit_val_batches: Union[float, int] = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(stage_name=stage_name, *args, **kwargs)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.checkpoint_metric = checkpoint_metric
        self.checkpoint_metric_mode = checkpoint_metric_mode
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

    def get_callbacks(self) -> List[L.Callback]:
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
            dirpath=self.checkpoint_dir,
            save_last=True,
        )

        performance_log = PerformanceLog()

        rich_progress_bar = RichProgressBar(
            leave=False, console_kwargs={"soft_wrap": True}
        )

        return [checkpoint_callback, rich_progress_bar, performance_log]

    def get_trainer(
        self, logger: Logger, callbacks: List[L.Callback]
    ) -> L.Trainer:
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
        return L.Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            num_nodes=self.num_nodes,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            log_every_n_steps=self.log_every_n_steps,
        )

    def run_model(
        self,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ):
        print(f"Training will start")
        print(f"\tExperiment path: {self.experiment_dir}")
        result = trainer.fit(model, data_module)

        print(f"Training finished")
        print(f"Last checkpoint saved at: {self.checkpoint_dir}/last.ckpt")
        return result


class LightningTest(LightningExperiment):
    _STAGE_NAME="test"
    
    def __init__(self, limit_test_batches: Union[float, int] = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit_test_batches = limit_test_batches
    
    def get_callbacks(self) -> List[L.Callback]:
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

    def get_trainer(
        self, logger: Logger, callbacks: List[L.Callback]
    ) -> L.Trainer:
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
            logger=logger,
            callbacks=callbacks,
            accelerator=self.accelerator,
            devices=self.devices,
            num_nodes=self.num_nodes,
            limit_test_batches=self.limit_test_batches,
            log_every_n_steps=self.log_every_n_steps
        )
        return trainer

    def run_model(
        self,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ) -> Any:
        return trainer.test(model, data_module)


class LightningSSLTrain(LightningTrain):
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
        super().__init__(stage_name=training_mode,*args, **kwargs)
        self.training_mode = training_mode
        self.load_backbone = load_backbone
        assert self.training_mode in ["pretrain", "finetune"]

    def get_model(self) -> L.LightningModule:
        """Get the model to use for the experiment.

        Returns
        -------
        L.LightningModule
            The model to use for the experiment
        """
        if self.training_mode == "pretrain":
            return self.get_pretrain_model()
        else:
            return self.get_finetune_model(self.load_backbone)

    def get_data_module(self) -> L.LightningDataModule:
        if self.training_mode == "pretrain":
            return self.get_pretrain_data_module()
        else:
            return self.get_finetune_data_module()

    @abstractmethod
    def get_pretrain_model(self) -> L.LightningModule:
        """Get the model to use for the pretraining phase.

        Returns
        -------
        L.LightningModule
            The model to use for the pretraining phase
        """
        raise NotImplementedError

    @abstractmethod
    def get_finetune_model(
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

    @abstractmethod
    def get_pretrain_data_module(self) -> L.LightningDataModule:
        """The data module to use for pre-training.

        Returns
        -------
        L.LightningDataModule
            The data module to use for pre-training
        """
        raise NotImplementedError

    @abstractmethod
    def get_finetune_data_module(self) -> L.LightningDataModule:
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

