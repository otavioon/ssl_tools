from typing import Union
import logging
import os
from datetime import datetime

EXPERIMENT_VERSION_FORMAT = "%Y-%m-%d_%H-%M-%S"


class LightningTrain:
    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 1e-3,
        log_dir: str = "logs",
        name: str = None,
        version: Union[str, int] = None,
        load: str = None,
        checkpoint_metric: str = None,
        checkpoint_metric_mode: str = "min",
        accelerator: str = "cpu",
        devices: int = 1,
        strategy: str = "auto",
        limit_train_batches: Union[float, int] = 1.0,
        limit_val_batches: Union[float, int] = 1.0,
        num_nodes: int = 1,
        num_workers: int = None,
        seed: int = None,
    ):
        """Defines the parameters for training a Lightning model. This class
        may be used to define the parameters for a Lightning experiment and
        CLI.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to pre-train the model
        batch_size : int, optional
            The batch size
        learning_rate : float, optional
            The learning rate of the optimizer
        log_dir : str, optional
            Path to the location where logs will be stored
        name: str, optional
            The name of the experiment (it will be used to compose the path of
            the experiments, such as logs and checkpoints)
        version: Union[int, str], optional
            The version of the experiment. If not is provided the current date
            and time will be used as the version
        load: str, optional
            The path to a checkpoint to load
        checkpoint_metric: str, optional
            The metric to monitor for checkpointing. If not provided, the last
            model will be saved
        checkpoint_metric_mode: str, optional
            The mode of the metric to monitor (min, max or mean). Defaults to
            "min"
        accelerator: str, optional
            The accelerator to use. Defaults to "cpu"
        devices: int, optional
            The number of devices to use. Defaults to 1
        strategy: str, optional
            The strategy to use. Defaults to "auto"
        limit_train_batches: Union[float, int], optional
            The number of batches to use for training. Defaults to 1.0 (use
            all batches)
        limit_val_batches: Union[float, int], optional
            The number of batches to use for validation. Defaults to 1.0 (use
            all batches)
        num_nodes: int, optional
            The number of nodes to use. Defaults to 1
        num_workers: int, optional
            The number of workers to use for the dataloader.
        seed: int, optional
            The seed to use.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.experiment_name = name
        self.experiment_version = version or datetime.now().strftime(
            EXPERIMENT_VERSION_FORMAT
        )
        self.load = load
        self.checkpoint_metric = checkpoint_metric
        self.checkpoint_metric_mode = checkpoint_metric_mode
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.seed = seed


class LightningTest:
    def __init__(
        self,
        load: str,
        batch_size: int = 1,
        log_dir="logs",
        name: str = None,
        version: str = None,
        accelerator: str = "cpu",
        devices: int = 1,
        limit_test_batches: Union[float, int] = 1.0,
        num_nodes: int = 1,
        num_workers: int = None,
        seed: int = None,
    ):
        """Defines the parameters for testing a Lightning model. This class
        may be used to define the parameters for a Lightning experiment and
        CLI.

        Parameters
        ----------
        load : str
            Path to the checkpoint to load
        batch_size : int, optional
            The batch size
        log_dir : str, optional
            Path to the location where logs will be stored
        name: str, optional
            The name of the experiment (it will be used to compose the path of
            the experiments, such as logs and checkpoints)
        version: Union[int, str], optional
            The version of the experiment. If not is provided the current date
            and time will be used as the version
        accelerator: str, optional
            The accelerator to use. Defaults to "cpu"
        devices: int, optional
            The number of devices to use. Defaults to 1
        limit_test_batches : Union[float, int], optional
            Limit the number of batches to use for testing.
        num_nodes: int, optional
            The number of nodes to use. Defaults to 1
        num_workers: int, optional
            The number of workers to use for the dataloader.
        seed: int, optional
            The seed to use.
        """
        self.load = load
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.experiment_name = name
        self.experiment_version = version or datetime.now().strftime(
            EXPERIMENT_VERSION_FORMAT
        )
        self.accelerator = accelerator
        self.devices = devices
        self.limit_test_batches = limit_test_batches
        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.seed = seed
