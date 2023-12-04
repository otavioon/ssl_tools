#!/usr/bin/env python

# TODO: A way of removing the need to add the path to the root of
# the project
from pathlib import Path
import sys


sys.path.append("../../../")


import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from datetime import datetime
from jsonargparse import CLI

from ssl_tools.models.layers import GRUEncoder
from ssl_tools.data.data_modules import (
    MultiModalHARDataModule,
    TNCHARDataModule,
    TFCDataModule,
    HARDataModule,
)
from ssl_tools.apps import LightningTrainCLI
from ssl_tools.losses.nxtent import NTXentLoss_poly
from ssl_tools.models.layers.linear import Discriminator
from ssl_tools.callbacks.performance import PerformanceLog
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from torchmetrics import Accuracy


class LightningTrainCLI(LightningTrainCLI):
    """Defines a Main CLI for (pre-)training Self-supervised Pytorch Lightning
    models

    Model and its speficic parameters are defined inner functions. Any inner
    function is callable from the command line and its parameters are exposed
    as command line arguments. Functions with names beginning with an underscore
    are not callable from the command line.

    In general, the train of models is done as follows:
        1. Assert the validity of the parameters
        2. Set the experiment name and version
        3. Instantiate the model (load from checkpoint if `load` is provided)
        4. Instantiate the data modules
        5. Instantiate trainer specific resources (logger, callbacks, etc.)
        6. Log the hyperparameters (for reproducibility purposes)
        7. Instantiate the trainer
        8. Train the model
    """

    def __init__(
        self,
        data: str,
        training_mode: str = "pretrain",
        load_backbone: str = None,
        *args,
        **kwargs,
    ):
        """Defines a Main CLI for (pre-)training Self-supervised Pytorch

        Parameters
        ----------
        data : str
            The location of the data
        training_mode : str, optional
            The training mode ("pretrain" or "finetune"), by default "pretrain"
        load_pretrain : str, optional
            Path to load the pre-trained backbone. The ``load`` parameter loads
            the whole model for downstream task, while this parameter only
            loads the backbone.
        """

        assert training_mode in ["pretrain", "finetune"], (
            f"training_mode must be either 'pretrain' or 'finetune'. "
            + f"Got {training_mode}"
        )

        super().__init__(*args, **kwargs)
        self.data = data
        self.training_mode = training_mode
        self.load_backbone = load_backbone
        self.checkpoint_path = None

    def _set_experiment(self, model_name: str):
        # Set the experiment variables (name, version and checkpoint path)
        self.experiment_name = self.experiment_name or model_name
        self.experiment_version = (
            self.experiment_version or datetime.now().strftime("%Y%m%d.%H%M%S")
        )
        self.checkpoint_path = (
            Path(self.log_dir)
            / self.experiment_name
            / self.experiment_version
            / "checkpoints"
        )
        # Defines the seed for reproducibility
        if self.seed:
            L.seed_everything(self.seed)

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

    def _train(self, model: L.LightningModule, data_module, trainer: L.Trainer):
        print(
            f"Start training. \n"
            + f"\tExperiment: {self.experiment_name} \n"
            + f"\tVersion is {self.experiment_version}"
        )
        return trainer.fit(model, data_module, ckpt_path=self.resume)

    def _load_model(self, model: L.LightningModule, path: str):
        print(f"Loading model from: {path}")
        state_dict = torch.load(path)["state_dict"]
        model.load_state_dict(state_dict)
        print("Model loaded successfully")

    def cpc(
        self,
        encoding_size: int = 150,
        window_size: int = 4,
        pad_length: bool = False,
        num_classes: int = 6,
        update_backbone: bool = False,
    ):
        """Trains the constrastive predictive coding model

        Parameters
        ----------
        encoding_size : int, optional
            Size of the encoding (output of the linear layer)
        window_size : int, optional
            Size of the input windows (X_t) to be fed to the encoder
        pad_length : bool, optional
            If True, the samples are padded to the length of the longest sample
            in the dataset.
        num_classes : int, optional
            Number of classes in the dataset. Only used in finetune mode.
        update_backbone : bool, optional
            If True, the backbone will be updated during training. Only used in
            finetune mode.
        """
        from ssl_tools.models.ssl import CPC
        from ssl_tools.models.ssl.classifier import SSLDiscriminator
        from ssl_tools.models.layers.linear import StateClassifier

        # ----------------------------------------------------------------------
        # 1. Assert the validity of the parameters
        # ----------------------------------------------------------------------
        if self.training_mode == "pretrain":
            assert self.batch_size == 1, (
                "CPC only supports batch size of 1. Please set batch_size=1"
            )
        # ----------------------------------------------------------------------
        # 2. Set experiment name and version
        # ----------------------------------------------------------------------
        self._set_experiment(f"CPC_{self.training_mode}")

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

        if self.training_mode == "finetune":
            if self.load_backbone:
                self._load_model(model, self.load_backbone)

            classifier = StateClassifier(
                input_size=encoding_size,
                n_classes=num_classes,
            )

            task = "multiclass" if num_classes > 2 else "binary"
            model = SSLDiscriminator(
                backbone=model,
                head=classifier,
                loss_fn=torch.nn.CrossEntropyLoss(),
                learning_rate=self.learning_rate,
                metrics={"acc": Accuracy(task=task, num_classes=num_classes)},
                update_backbone=update_backbone,
            )

        if self.load:
            self._load_model(model, self.load)

        # ----------------------------------------------------------------------
        # 4. Instantiate data modules
        # ----------------------------------------------------------------------
        if self.training_mode == "pretrain":
            data_module = MultiModalHARDataModule(
                self.data,
                batch_size=self.batch_size,
                fix_length=pad_length,
                num_workers=self.num_workers,
            )
        else:
            data_module = HARDataModule(
                self.data,
                batch_size=self.batch_size,
                label="standard activity code",
                features_as_channels=True,
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
        w: float = 0.05,
        significance_level: float = 0.01,
        repeat: int = 5,
        pad_length: bool = True,
        num_classes: int = 6,
        update_backbone: bool = False,
    ):
        """Trains the Temporal Neighborhood Coding model

        Parameters
        ----------
        encoding_size : int, optional
            Size of the encoding (output of the linear layer) .
        window_size : int, optional
            Size of the input windows (X_t) to be fed to the encoder.
        mc_sample_size : int
            The number of close and distant samples selected in the dataset.
        w : float
            This parameter is used in loss and represent probability of
            sampling a positive window from the non-neighboring region.
        significance_level: float, optional
            The significance level of the ADF test. It is used to reject the
            null hypothesis of the test if p-value is less than this value.
        repeat : int, optional
            Simple repeat the element of the dataset ``repeat`` times.
        pad_length : bool, optional
            If True, the samples are padded to the length of the longest sample
            in the dataset.
        num_classes : int, optional
            Number of classes in the dataset. Only used in finetune mode.
        update_backbone : bool, optional
            If True, the backbone will be updated during training. Only used in
            finetune mode.
        """
        from ssl_tools.models.ssl import TNC
        from ssl_tools.models.ssl.classifier import SSLDiscriminator
        from ssl_tools.models.layers.linear import StateClassifier

        # ----------------------------------------------------------------------
        # 1. Assert the validity of the parameters
        # ----------------------------------------------------------------------
        assert (
            significance_level > 0 and significance_level < 1
        ), "The significance level must be between 0 and 1"

        # ----------------------------------------------------------------------
        # 2. Set experiment name and version
        # ----------------------------------------------------------------------
        self._set_experiment(f"TNC_{self.training_mode}")

        # ----------------------------------------------------------------------
        # 3. Instantiate model
        # ----------------------------------------------------------------------
        discriminator = Discriminator(input_size=encoding_size)
        encoder = GRUEncoder(encoding_size=encoding_size)
        model = TNC(
            discriminator=discriminator,
            encoder=encoder,
            mc_sample_size=mc_sample_size,
            w=w,
            learning_rate=self.learning_rate,
        )
        
        if self.training_mode == "finetune":
            if self.load_backbone:
                self._load_model(model, self.load_backbone)
            
            classifier = StateClassifier(
                input_size=encoding_size,
                n_classes=num_classes,
            )         
            
            task = "multiclass" if num_classes > 2 else "binary"
            model = SSLDiscriminator(
                backbone=model,
                head=classifier,
                loss_fn=torch.nn.CrossEntropyLoss(),
                learning_rate=self.learning_rate,
                metrics={"acc": Accuracy(task=task, num_classes=num_classes)},
                update_backbone=update_backbone,
            )   
            
        if self.load:
            self._load_model(model, self.load)

        # ----------------------------------------------------------------------
        # 4. Instantiate data modules
        # ----------------------------------------------------------------------
        if self.training_mode == "pretrain":
            data_module = TNCHARDataModule(
                self.data,
                batch_size=self.batch_size,
                fix_length=pad_length,
                window_size=window_size,
                mc_sample_size=mc_sample_size,
                significance_level=significance_level,
                repeat=repeat,
                num_workers=self.num_workers,
            )
        else:
            data_module = HARDataModule(
                self.data,
                batch_size=self.batch_size,
                label="standard activity code",
                features_as_channels=True,
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
        label: str = "standard activity code",
        features_as_channels: bool = True,
        jitter_ratio: float = 2,
        num_classes: int = 6,
        update_backbone: bool = False,
    ):
        """Trains the Temporal Frequency Coding model

        Parameters
        ----------
        length_alignment : int, optional
            Truncate the features to this value.
        use_cosine_similarity : bool, optional
            If True use cosine similarity, otherwise use dot product in the
            NXTent loss.
        temperature : float, optional
            Temperature parameter of the NXTent loss.
        label : str, optional
            Name of the column with the labels.
        features_as_channels : bool, optional
            If true, features will be transposed to (C, T), where C is the
            number of features and T is the number of time steps. If False,
            features will be (T*C, )
        jitter_ratio : float, optional
            Ratio of the standard deviation of the gaussian noise that will be
            added to the data.
        num_classes : int, optional
            Number of classes in the dataset. Only used in finetune mode.
        update_backbone : bool, optional
            If True, the backbone will be updated during training. Only used in
            finetune mode.
        """
        from ssl_tools.models.ssl import TFC
        from ssl_tools.models.ssl.classifier import SSLDiscriminator
        from ssl_tools.models.layers.linear import SimpleClassifier

        # ----------------------------------------------------------------------
        # 1. Assert the validity of the parameters
        # ----------------------------------------------------------------------
        pass

        # ----------------------------------------------------------------------
        # 2. Set experiment name and version
        # ----------------------------------------------------------------------
        self._set_experiment(f"TFC_{self.training_mode}")

        # ----------------------------------------------------------------------
        # 3. Instantiate model
        # ----------------------------------------------------------------------
        time_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                length_alignment, dim_feedforward=2 * length_alignment, nhead=2
            ),
            num_layers=2,
        )
        frequency_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                length_alignment, dim_feedforward=2 * length_alignment, nhead=2
            ),
            num_layers=2,
        )

        time_projector = torch.nn.Sequential(
            torch.nn.Linear(length_alignment, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        frequency_projector = torch.nn.Sequential(
            torch.nn.Linear(length_alignment, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )

        nxtent = NTXentLoss_poly(
            temperature=temperature,
            use_cosine_similarity=use_cosine_similarity,
        )

        model = TFC(
            time_encoder=time_encoder,
            frequency_encoder=frequency_encoder,
            time_projector=time_projector,
            frequency_projector=frequency_projector,
            nxtent_criterion=nxtent,
            learning_rate=self.learning_rate,
        )
        
        if self.training_mode == "finetune":
            if self.load_backbone:
                self._load_model(model, self.load_backbone)
                
            classifier = SimpleClassifier(
                input_size=2*128,
                num_classes=num_classes,
            )
            
            task = "multiclass" if num_classes > 2 else "binary"
            model = SSLDiscriminator(
                backbone=model,
                head=classifier,
                loss_fn=torch.nn.CrossEntropyLoss(),
                learning_rate=self.learning_rate,
                metrics={"acc": Accuracy(task=task, num_classes=num_classes)},
                update_backbone=update_backbone,
            )
            
        if self.load:
            self._load_model(model, self.load)

        # ----------------------------------------------------------------------
        # 4. Instantiate data modules
        # ----------------------------------------------------------------------
        if self.training_mode == "pretrain":
            data_module = TFCDataModule(
                self.data,
                batch_size=self.batch_size,
                label=label,
                features_as_channels=features_as_channels,
                length_alignment=length_alignment,
                time_transforms=None,  # None, use default transforms.
                # Check TFCDataModule for details
                frequency_transforms=None,  # None, use default transforms
                # Check TFCDataModule for details
                jitter_ratio=jitter_ratio,
                num_workers=self.num_workers,
            )
        else:
            data_module = HARDataModule(
                self.data,
                batch_size=self.batch_size,
                label="standard activity code",
                features_as_channels=features_as_channels,
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


if __name__ == "__main__":
    CLI(LightningTrainCLI, as_positional=False)
