#!/usr/bin/env python

import lightning as L
import torch

from ssl_tools.experiments import LightningSSLTrain, LightningTest, auto_main
from torchmetrics import Accuracy
from ssl_tools.models.ssl.classifier import SSLDiscriminator
from ssl_tools.models.ssl.modules.heads import TFCPredictionHead
from ssl_tools.models.ssl.tfc import build_tfc_transformer
from ssl_tools.data.data_modules import TFCDataModule


class TFCTrain(LightningSSLTrain):
    _MODEL_NAME = "TFC"

    def __init__(
        self,
        data: str,
        label: str = "standard activity code",
        encoding_size: int = 128,
        in_channels: int = 6,
        length_alignment: int = 178,
        use_cosine_similarity: bool = True,
        temperature: float = 0.5,
        features_as_channels: bool = False,
        jitter_ratio: float = 2,
        num_classes: int = 6,
        update_backbone: bool = False,
        *args,
        **kwargs,
    ):
        """Trains the Temporal Frequency Coding model

        Parameters
        ----------
        encoding_size : int, optional
            Size of the encoding (output of the linear layer). Note that the
            representation will be of size 2*encoding_size, since the
            representation is the concatenation of the time and frequency
            encodings.
        label : str, optional
            Name of the column with the labels.
        encoding_size : int, optional
            Size of the encoding (output of the linear layer). The real size of
            the representation will be 2*encoding_size, since the
            representation is the concatenation of the time and frequency
            encodings.
        in_channels : int, optional
            Number of channels in the input data
        length_alignment : int, optional
            Truncate the features to this value.
        use_cosine_similarity : bool, optional
            If True use cosine similarity, otherwise use dot product in the
            NXTent loss.
        temperature : float, optional
            Temperature parameter of the NXTent loss.
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
        super().__init__(*args, **kwargs)
        self.data = data
        self.label = label
        self.encoding_size = encoding_size
        self.in_channels = in_channels
        self.length_alignment = length_alignment
        self.use_cosine_similarity = use_cosine_similarity
        self.temperature = temperature
        self.features_as_channels = features_as_channels
        self.jitter_ratio = jitter_ratio
        self.num_classes = num_classes
        self.update_backbone = update_backbone

    def get_pretrain_model(self) -> L.LightningModule:
        model = build_tfc_transformer(
            encoding_size=self.encoding_size,
            in_channels=self.in_channels,
            length_alignment=self.length_alignment,
            use_cosine_similarity=self.use_cosine_similarity,
            temperature=self.temperature,
            learning_rate=self.learning_rate,
        )
        return model

    def get_pretrain_data_module(self) -> L.LightningDataModule:
        data_module = TFCDataModule(
            self.data,
            batch_size=self.batch_size,
            label=self.label,
            features_as_channels=self.features_as_channels,
            length_alignment=self.length_alignment,
            jitter_ratio=self.jitter_ratio,
            num_workers=self.num_workers,
            only_time_frequency=False,
        )
        return data_module

    def get_finetune_model(
        self, load_backbone: str = None
    ) -> L.LightningModule:
        model = self.get_pretrain_model()

        if load_backbone is not None:
            self.load_checkpoint(model, load_backbone)

        classifier = TFCPredictionHead(
            input_dim=2 * self.encoding_size,
            output_dim=self.num_classes,
        )

        task = "multiclass" if self.num_classes > 2 else "binary"
        model = SSLDiscriminator(
            backbone=model,
            head=classifier,
            loss_fn=torch.nn.CrossEntropyLoss(),
            learning_rate=self.learning_rate,
            metrics={"acc": Accuracy(task=task, num_classes=self.num_classes)},
            update_backbone=self.update_backbone,
        )
        return model

    def get_finetune_data_module(self) -> L.LightningDataModule:
        data_module = TFCDataModule(
            self.data,
            batch_size=self.batch_size,
            label=self.label,
            features_as_channels=self.features_as_channels,
            length_alignment=self.length_alignment,
            jitter_ratio=self.jitter_ratio,
            num_workers=self.num_workers,
            only_time_frequency=True,
        )
        return data_module


class TFCTest(LightningTest):
    _MODEL_NAME = "TFC"

    def __init__(
        self,
        data: str,
        label: str = "standard activity code",
        encoding_size: int = 128,
        in_channels: int = 6,
        length_alignment: int = 178,
        use_cosine_similarity: bool = True,
        temperature: float = 0.5,
        features_as_channels: bool = False,
        num_classes: int = 6,
        *args,
        **kwargs,
    ):
        """Tests the Temporal Frequency Coding model

        Parameters
        ----------
        encoding_size : int, optional
            Size of the encoding (output of the linear layer). Note that the
            representation will be of size 2*encoding_size, since the
            representation is the concatenation of the time and frequency
            encodings.
        label : str, optional
            Name of the column with the labels.
        encoding_size : int, optional
            Size of the encoding (output of the linear layer). The real size of
            the representation will be 2*encoding_size, since the
            representation is the concatenation of the time and frequency
            encodings.
        in_channels : int, optional
            Number of channels in the input data
        length_alignment : int, optional
            Truncate the features to this value.
        use_cosine_similarity : bool, optional
            If True use cosine similarity, otherwise use dot product in the
            NXTent loss.
        temperature : float, optional
            Temperature parameter of the NXTent loss.
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
        super().__init__(*args, **kwargs)
        self.data = data
        self.label = label
        self.encoding_size = encoding_size
        self.in_channels = in_channels
        self.length_alignment = length_alignment
        self.use_cosine_similarity = use_cosine_similarity
        self.temperature = temperature
        self.features_as_channels = features_as_channels
        self.num_classes = num_classes

    def get_model(self) -> L.LightningModule:
        model = build_tfc_transformer(
            encoding_size=self.encoding_size,
            in_channels=self.in_channels,
            length_alignment=self.length_alignment,
            use_cosine_similarity=self.use_cosine_similarity,
            temperature=self.temperature,
        )

        classifier = TFCPredictionHead(
            input_dim=2 * self.encoding_size,
            output_dim=self.num_classes,
        )

        task = "multiclass" if self.num_classes > 2 else "binary"
        model = SSLDiscriminator(
            backbone=model,
            head=classifier,
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics={"acc": Accuracy(task=task, num_classes=self.num_classes)},
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = TFCDataModule(
            self.data,
            batch_size=self.batch_size,
            label=self.label,
            features_as_channels=self.features_as_channels,
            length_alignment=self.length_alignment,
            num_workers=self.num_workers,
            only_time_frequency=True,
        )
        return data_module


if __name__ == "__main__":
    options = {
        "fit": TFCTrain,
        "test": TFCTest,
    }
    auto_main(options)
