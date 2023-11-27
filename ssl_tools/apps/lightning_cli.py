from typing import Union


class LightningTrainCLI:
    def __init__(
        self,
        data: str,
        epochs: int = 1,
        batch_size: int = 1,
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
        """Defines a Main CLI for pre-training Pytorch Lightning models

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
