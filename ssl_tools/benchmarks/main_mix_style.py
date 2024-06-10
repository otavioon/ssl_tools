#!/usr/bin/env python3
import traceback
from dassl.modeling.ops.mixstyle import (
    MixStyle,
    run_without_mixstyle,
    run_with_mixstyle,
    random_mixstyle,
    deactivate_mixstyle,
)
from dassl.modeling.ops import Conv2dDynamic

import torch
from sklearn.manifold import TSNE
from ssl_tools.data.data_modules.har import MultiModalHARSeriesDataModule
import lightning as L
from ssl_tools.models.nets.simple import SimpleClassificationNet
from lightning.pytorch.cli import LightningCLI
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from torchmetrics import Accuracy
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

import numpy as np
import time


from functools import partial
from typing import Literal, Tuple
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchmetrics import Accuracy
from ssl_tools.models.nets.simple import SimpleClassificationNet
from ssl_tools.models.utils import RandomDataModule
import lightning as L

from ssl_tools.models.utils import ZeroPadder2D

import ray

from ssl_tools.transforms.utils import Cast


################################################################################
# Model code
################################################################################
class SimpleClassificationNet2(SimpleClassificationNet):
    def single_step(self, batch: torch.Tensor, batch_idx: int, step_name: str):
        return super().single_step(batch[:2], batch_idx, step_name)


# ******************************************
# CNN_HaEtAl_1D
# ******************************************


class CNN_HaEtAl_1D_Backbone(torch.nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.input_channels = input_channels

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(1, 4),
            stride=(1, 1),
        )
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size=(1, 3),
            stride=(1, 3),
        )
        self.mixstyle1 = MixStyle(0.5, 0.1, mix="random")

        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, 5),
            stride=(1, 1),
        )
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(
            kernel_size=(1, 3),
            stride=(1, 3),
        )
        self.mixstyle2 = MixStyle(0.5, 0.1, mix="random")

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.mixstyle1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.mixstyle2(x)

        return x


class CNN_HaEtAl_1D(SimpleClassificationNet2):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_shape=input_shape)
        self.fc_input_channels = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = self._create_fc(self.fc_input_channels, num_classes)
        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            val_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
            test_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
        )

    def _create_backbone(self, input_shape: Tuple[int, int]) -> torch.nn.Module:
        return CNN_HaEtAl_1D_Backbone(input_channels=input_shape[0])

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int, int]
            The input shape of the network.

        Returns
        -------
        int
            The number of features after the convolutional layers.
        """
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )


# ******************************************
# CNN_HaEtAl_2D
# ******************************************
def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv3x3_dynamic(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    attention_in_channels: int = None,
) -> Conv2dDynamic:
    """3x3 convolution with padding"""
    return Conv2dDynamic(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        attention_in_channels=attention_in_channels,
    )


class CNN_HaEtAl_2D_Backbone(torch.nn.Module):
    def __init__(self, pad_at: int, in_channels: int = 1):
        super().__init__()
        self.first_kernel_size = 4
        self.in_channels = in_channels
        self.pad_at = pad_at

        # Add padding
        self.zero_padder = ZeroPadder2D(
            pad_at=self.pad_at,
            padding_size=self.first_kernel_size - 1,  # kernel size - 1
        )
        # First 2D convolutional layer
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(self.first_kernel_size, self.first_kernel_size),
            stride=(1, 1),
        )
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(3, 3),
            padding=1,
        )
        self.mixstyle1 = MixStyle(0.5, 0.1, mix="random")

        # Second 2D convolutional layer
        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=2,
        )
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(3, 3),
            padding=1,
        )
        self.mixstyle2 = MixStyle(0.5, 0.1, mix="random")

    def forward(self, x):
        x = self.zero_padder(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.mixstyle1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.mixstyle2(x)
        return x


class CNN_HaEtAl_2D(SimpleClassificationNet2):
    def __init__(
        self,
        pad_at: List[int] = (3,),
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        self.pad_at = pad_at
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_shape=input_shape)
        self.fc_input_channels = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = self._create_fc(self.fc_input_channels, num_classes)
        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            val_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
            test_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
        )

    def _create_backbone(self, input_shape: Tuple[int, int]) -> torch.nn.Module:
        return CNN_HaEtAl_2D_Backbone(
            pad_at=self.pad_at, in_channels=input_shape[0]
        )

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int, int]
            The input shape of the network.

        Returns
        -------
        int
            The number of features after the convolutional layers.
        """
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )


# ******************************************
# Resnet1D
# ******************************************


class ConvolutionalBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, activation_cls: torch.nn.Module = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.activation_cls = activation_cls

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels, out_channels=64, kernel_size=5, stride=1
            ),
            torch.nn.BatchNorm1d(64),
            activation_cls(),
            torch.nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.block(x)


class SqueezeAndExcitation1D(torch.nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.num_channels_reduced = in_channels // reduction_ratio

        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_channels, self.num_channels_reduced),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_channels_reduced, in_channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        squeeze_tensor = input_tensor.mean(dim=2)
        x = self.block(squeeze_tensor)
        output_tensor = torch.mul(
            input_tensor,
            x.view(input_tensor.shape[0], input_tensor.shape[1], 1),
        )
        return output_tensor


class ResNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        activation_cls: torch.nn.Module = torch.nn.ReLU,
        mix_style_factor=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.activation_cls = activation_cls
        self.mix_style_factor = mix_style_factor

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            torch.nn.BatchNorm1d(32),
            activation_cls(),
            torch.nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            torch.nn.BatchNorm1d(64),
        )

        self.mix_style = MixStyle(self.mix_style_factor, 0.1, mix="random")

    def forward(self, x):
        input_tensor = x
        x = self.block(x)
        x += input_tensor
        x = self.activation_cls()(x)
        if self.mix_style_factor < 0.001:
            x = x.unsqueeze(1)
            x = self.mix_style(x)
            x = x.squeeze(1)
        return x


class ResNetSEBlock(ResNetBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block.append(SqueezeAndExcitation1D(64))


class _ResNet1D(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        residual_block_cls=ResNetBlock,
        activation_cls: torch.nn.Module = torch.nn.ReLU,
        num_residual_blocks: int = 5,
        reduction_ratio=2,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_residual_blocks = num_residual_blocks
        self.reduction_ratio = reduction_ratio

        self.conv_block = ConvolutionalBlock(
            in_channels=input_shape[0], activation_cls=activation_cls
        )
        self.residual_blocks = torch.nn.Sequential(
            *[
                residual_block_cls(
                    in_channels=64,
                    activation_cls=activation_cls,
                    mix_style_factor=factor,
                )
                for i, factor in zip(
                    range(num_residual_blocks), [0.5, 0.3, 0.2, 0, 0, 0, 0, 0]
                )
            ]
        )
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(2)
        return x


class ResNet1DBase(SimpleClassificationNet2):
    def __init__(
        self,
        resnet_block_cls: type = ResNetBlock,
        activation_cls: type = torch.nn.ReLU,
        input_shape: Tuple[int, int] = (6, 60),
        num_classes: int = 6,
        num_residual_blocks: int = 5,
        reduction_ratio=2,
        learning_rate: float = 1e-3,
    ):
        backbone = _ResNet1D(
            input_shape=input_shape,
            residual_block_cls=resnet_block_cls,
            activation_cls=activation_cls,
            num_residual_blocks=num_residual_blocks,
            reduction_ratio=reduction_ratio,
        )

        self.fc_input_features = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = torch.nn.Linear(self.fc_input_features, num_classes)

        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            val_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
            test_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
        )

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int, int]
            The input shape of the network.

        Returns
        -------
        int
            The number of features after the convolutional layers.
        """
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)


# Deep Residual Network for Smartwatch-Based User Identification through Complex Hand Movements (ResNet1D)
class ResNet1D_8(ResNet1DBase):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            resnet_block_cls=ResNetBlock,
            activation_cls=torch.nn.ELU,
            num_residual_blocks=8,
        )


# Deep Residual Network for Smartwatch-Based User Identification through Complex Hand Movements (ResNetSE1D)
class ResNetSE1D_8(ResNet1DBase):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            resnet_block_cls=ResNetSEBlock,
            activation_cls=torch.nn.ELU,
            num_residual_blocks=8,
        )


# resnet-se: Channel Attention-Based Deep Residual Network for Complex Activity Recognition Using Wrist-Worn Wearable Sensors
# Changes the activation function to ReLU and the number of residual blocks to 5 (compared to ResNetSE1D_8)
class ResNetSE1D_5(ResNet1DBase):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            resnet_block_cls=ResNetSEBlock,
            activation_cls=torch.nn.ReLU,
            num_residual_blocks=5,
        )


################################################################################
# Runner code
################################################################################


@dataclass
class ExperimentArgs:
    trainer_cls: Any
    model_cls: Any
    data_cls: Any
    trainer_args: Dict[str, Any]
    model_args: Dict[str, Any]
    train_data_args: Dict[str, Any]
    test_data_args: Dict[str, Any]
    seed: int = 42
    mix: bool = True


def pretty_print_experiment_args(args: ExperimentArgs, indent: int = 4) -> str:
    def repr_dict(d: Dict[str, Any], level: int) -> str:
        items = []
        for key, value in d.items():
            if isinstance(value, dict):
                items.append(
                    f"{' ' * (level * indent)}{key}={{"
                    + repr_dict(value, level + 1)
                    + "}"
                )
            else:
                items.append(f"{' ' * (level * indent)}{key}={value!r}")
        return ",\n".join(items)

    return (
        f"ExperimentArgs(\n"
        f"{' ' * indent}trainer_cls={args.trainer_cls.__name__},\n"
        f"{' ' * indent}model_cls={args.model_cls.__name__},\n"
        f"{' ' * indent}data_cls={args.data_cls.__name__},\n"
        f"{' ' * indent}trainer_args={{\n{repr_dict(args.trainer_args, 2)}\n{' ' * indent}}},\n"
        f"{' ' * indent}model_args={{\n{repr_dict(args.model_args, 2)}\n{' ' * indent}}},\n"
        f"{' ' * indent}train_data_args={{\n{repr_dict(args.train_data_args, 2)}\n{' ' * indent}}},\n"
        f"{' ' * indent}test_data_args={{\n{repr_dict(args.test_data_args, 2)}\n{' ' * indent}}},\n"
        f"{' ' * indent}seed={args.seed},\n"
        f"{' ' * indent}mix={args.mix}\n)"
    )


def cli_main(experiment: ExperimentArgs):
    print("*" * 80)
    print(f"Running experiment")
    print(pretty_print_experiment_args(experiment))
    print("*" * 80)

    class DummyModel(L.LightningModule):
        def __init__(self, *args, **kwargs):
            pass

    class DummyTrainer(L.Trainer):
        def __init__(self, *args, **kwargs):
            pass

    # Unpack experiment into a dict, ignoring the test_data for now
    cli_args = {
        "trainer": experiment.trainer_args,
        "model": experiment.model_args,
        "data": experiment.train_data_args,
        "seed_everything": experiment.seed,
    }

    # print(cli_args)

    # Instantiate model, trainer, and train_datamodule
    train_cli = LightningCLI(
        model_class=experiment.model_cls,
        datamodule_class=experiment.data_cls,
        trainer_class=experiment.trainer_cls,
        args=cli_args,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf"},
    )

    test_cli = LightningCLI(
        model_class=DummyModel,
        trainer_class=DummyTrainer,
        datamodule_class=experiment.data_cls,
        args={
            "trainer": {},
            "model": {},
            "data": experiment.test_data_args,
        },
        run=False,
    )

    # Shortcut to access the trainer, model and datamodule
    trainer = train_cli.trainer
    model = train_cli.model
    if experiment.mix:
        model.apply(random_mixstyle)
    else:
        model.apply(deactivate_mixstyle)

    train_data_module = train_cli.datamodule
    test_data_module = test_cli.datamodule

    # Attach model test metrics
    model.metrics["test"]["accuracy"] = Accuracy(
        task="multiclass", num_classes=7
    )

    # Perform FIT
    trainer.fit(model, train_data_module)

    # Perform test and return metrics
    metrics = trainer.test(model, test_data_module)
    return metrics


def _run_experiment_wrapper(experiment_args: ExperimentArgs):
    try:
        return cli_main(experiment_args)
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()
        raise e


def run_using_ray(experiments: List[ExperimentArgs], ray_address: str = None):
    print(f"Running {len(experiments)} experiments using RAY...")
    ray.init(address=ray_address)
    remotes_to_run = [
        ray.remote(
            num_gpus=0.25,
            num_cpus=4,
            max_calls=1,
            max_retries=0,
            retry_exceptions=False,
        )(_run_experiment_wrapper).remote(exp_args)
        for exp_args in experiments
    ]
    ready, not_ready = ray.wait(remotes_to_run, num_returns=len(remotes_to_run))
    print(f"Ready: {len(ready)}. Not ready: {len(not_ready)}")
    ray.shutdown()
    return ready, not_ready


def run_serial(experiments: List[ExperimentArgs]):
    print(f"Running {len(experiments)} experiments...")
    for exp_args in experiments:
        _run_experiment_wrapper(exp_args)


################################################################################
# Main code
################################################################################


def main_loo():
    datasets = [
        Path("/workspaces/hiaac-m4/data/standartized_balanced/KuHar"),
        Path("/workspaces/hiaac-m4/data/standartized_balanced/MotionSense"),
        Path("/workspaces/hiaac-m4/data/standartized_balanced/RealWorld_thigh"),
        Path("/workspaces/hiaac-m4/data/standartized_balanced/RealWorld_waist"),
        Path("/workspaces/hiaac-m4/data/standartized_balanced/UCI"),
        Path("/workspaces/hiaac-m4/data/standartized_balanced/WISDM"),
    ]

    run_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    print("*" * 80)
    print(f"Running Leave-One-Out experiment")
    print(f"Run ID: {run_id}")
    print("*" * 80)

    print()
    print()

    experiments = []

    for i, dataset in enumerate(datasets):
        train_datasets = datasets.copy()
        train_datasets.pop(i)

        experiment = ExperimentArgs(
            trainer_cls=L.Trainer,
            trainer_args={
                "max_epochs": 100,
                "accelerator": "gpu",
                "devices": 1,
                "logger": {
                    "class_path": "lightning.pytorch.loggers.CSVLogger",
                    "init_args": {
                        "save_dir": "logs/CNN_HaEtAl_2D/",
                        "name": dataset.stem,
                        "version": run_id,
                    },
                },
                "callbacks": [
                    {
                        "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                        "init_args": {"monitor": "val_loss", "patience": 20},
                    },
                    {
                        "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "init_args": {
                            "monitor": "val_loss",
                            "mode": "min",
                            "save_last": True,
                        },
                    },
                ],
            },
            model_cls=CNN_HaEtAl_2D,
            model_args={
                "input_shape": (1, 6, 60),
                "num_classes": 7,
                "learning_rate": 1e-3,
            },
            data_cls=MultiModalHARSeriesDataModule,
            train_data_args={
                "data_path": train_datasets,
                "batch_size": 128,
                "num_workers": 16,
                "transforms": [
                    {
                        "class_path": "ssl_tools.transforms.utils.Unsqueeze",
                        "init_args": {"axis": 0},
                    },
                    {
                        "class_path": "ssl_tools.transforms.utils.Cast",
                        "init_args": {"dtype": "float32"},
                    },
                ],
                "domain_info": True,
            },
            test_data_args={
                "data_path": dataset,
                "batch_size": 128,
                "num_workers": 8,
                "transforms": [
                    {
                        "class_path": "ssl_tools.transforms.utils.Unsqueeze",
                        "init_args": {"axis": 0},
                    },
                    {
                        "class_path": "ssl_tools.transforms.utils.Cast",
                        "init_args": {"dtype": "float32"},
                    },
                ],
                "domain_info": True,
            },
            seed=42,
            mix=True,
        )

        experiments.append(experiment)
        # cli_main(experiment)

    # run_serial(experiments)
    run_using_ray(experiments)


if __name__ == "__main__":
    main_loo()
