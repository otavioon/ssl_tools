#!/usr/bin/env python

# TODO: A way of removing the need to add the path to the root of
# the project
import sys
from jsonargparse import CLI
import lightning as L
import torch

sys.path.append("../../../")


from ssl_tools.apps import SSLTrain, SSLTest
from ssl_tools.models.ssl.cpc import build_cpc
from ssl_tools.data.data_modules import (
    MultiModalHARDataModule,
    HARDataModule,
)
from torchmetrics import Accuracy
from ssl_tools.models.ssl.classifier import SSLDiscriminator
from ssl_tools.models.ssl.cpc import CPCHead


class CPCTrain(SSLTrain):
    _MODEL_NAME = "CPC"

    def __init__(
        self,
        data: str,
        encoding_size: int = 150,
        in_channel: int = 6,
        window_size: int = 4,
        pad_length: bool = False,
        num_classes: int = 6,
        update_backbone: bool = False,
        *args,
        **kwargs,
    ):
        """Trains the constrastive predictive coding model

        Parameters
        ----------
        encoding_size : int, optional
            Size of the encoding (output of the linear layer)
        in_channel : int, optional
            Number of channels in the input data
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
        super().__init__(*args, **kwargs)
        self.data = data
        self.encoding_size = encoding_size
        self.in_channel = in_channel
        self.window_size = window_size
        self.pad_length = pad_length
        self.num_classes = num_classes
        self.update_backbone = update_backbone

    def _get_pretrain_model(self) -> L.LightningModule:
        model = build_cpc(
            encoding_size=self.encoding_size,
            in_channel=self.in_channel,
            learning_rate=self.learning_rate,
            window_size=self.window_size,
            n_size=5,
        )
        return model

    def _get_pretrain_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARDataModule(
            self.data,
            batch_size=self.batch_size,
            fix_length=self.pad_length,
            num_workers=self.num_workers,
        )
        return data_module

    def _get_finetune_model(
        self, load_backbone: str = None
    ) -> L.LightningModule:
        model = self._get_pretrain_model()

        if load_backbone is not None:
            self._load_model(model, load_backbone)

        classifier = CPCHead(
            input_size=self.encoding_size,
            n_classes=self.num_classes,
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

    def _get_finetune_data_module(self) -> L.LightningDataModule:
        data_module = HARDataModule(
            self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
        )

        return data_module
    
class CPCTest(SSLTest):
    _MODEL_NAME = "CPC"

    def __init__(
        self,
        data: str,
        encoding_size: int = 150,
        in_channel: int = 6,
        window_size: int = 4,
        num_classes: int = 6,
        *args,
        **kwargs,
    ):
        """Trains the constrastive predictive coding model

        Parameters
        ----------
        encoding_size : int, optional
            Size of the encoding (output of the linear layer)
        in_channel : int, optional
            Number of channels in the input data
        window_size : int, optional
            Size of the input windows (X_t) to be fed to the encoder
        num_classes : int, optional
            Number of classes in the dataset. Only used in finetune mode.
        update_backbone : bool, optional
            If True, the backbone will be updated during training. Only used in
            finetune mode.
        """
        super().__init__(*args, **kwargs)
        self.data = data
        self.encoding_size = encoding_size
        self.in_channel = in_channel
        self.window_size = window_size
        self.num_classes = num_classes

    def _get_test_model(
        self, load_backbone: str = None
    ) -> L.LightningModule:
        model = build_cpc(
            encoding_size=self.encoding_size,
            in_channel=self.in_channel,
            window_size=self.window_size,
            n_size=5,
        )

        classifier = CPCHead(
            input_size=self.encoding_size,
            n_classes=self.num_classes,
        )

        task = "multiclass" if self.num_classes > 2 else "binary"
        model = SSLDiscriminator(
            backbone=model,
            head=classifier,
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics={"acc": Accuracy(task=task, num_classes=self.num_classes)},
        )
        return model

    def _get_test_data_module(self) -> L.LightningDataModule:
        data_module = HARDataModule(
            self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
        )

        return data_module


def main():
    components = {
        "fit": CPCTrain,
        "test": CPCTest,
    }
    CLI(components=components, as_positional=False)()


if __name__ == "__main__":
    main()