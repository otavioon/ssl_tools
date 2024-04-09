from operator import attrgetter
from typing import List, Tuple

from lightning import LightningModule, Trainer
from torchmetrics import Accuracy
from ssl_tools.utils.layers import OutputLoggerCallback
from functools import partial
import numpy as np
import plotly.express as px
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from collections import defaultdict


# class OutputLoggerCallback(L.Callback):
#     def __init__(self, layers: List[str]):
#         self.layers = layers
#         self._layer_results = defaultdict(list)
#         self._handlers = {}

#     def count(self, module, input, output, layer_name: str):
#         self._layer_results[layer_name].append(output.detach().cpu())

#     def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
#         if stage != "test":
#             return

#         for layer_name in self.layers:
#             layer = attrgetter(layer_name)(pl_module)
#             handle = layer.register_forward_hook(partial(self.count, layer_name=layer_name))
#             self._handlers[layer_name] = handle


#     def teardown(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
#         for layer_name, handle in self._handlers.items():
#             handle.remove()


# class LatentAnalysisCallback(OutputLoggerCallback):
#     def __init__(self, layers: List[str], sklearn_technique_cls, **sklearn_technique_kwargs):
#         super().__init__(layers)
#         self.sklearn_cls = partial(sklearn_technique_cls, **sklearn_technique_kwargs)

#     def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
#         if stage != "test":
#             return

#         for layer_name, outputs in self._layer_results.items():
#             # concatenate the outputs
#             outputs = torch.cat(outputs, dim=0)
#             self._layer_results[layer_name] = outputs

#             outputs = outputs.numpy()
#             outputs = outputs.reshape(outputs.shape[0], -1)

#             # instantiate the sklearn model
#             sklearn_model = self.sklearn_cls()

#             print(f"Fitting a T-SNE for {layer_name}...")
#             transformed_outputs = sklearn_model.fit_transform(outputs)

#             output_file = f"{layer_name}_transformed"
#             np.save(f"{output_file}.npy", transformed_outputs)
#             print(f"Saved transformed data to {output_file}")

#             fig = px.scatter(transformed_outputs, x=0, y=1)
#             fig.write_image(f"{output_file}.png")
#             print(f"Saved plot to {output_file}.png")


#         super().teardown(trainer, pl_module, stage)


# class LatentAnalysis(L.LightningModule):
#     def __init__(self, model: L.LightningModule, layers: List[str]):
#         self.layer_output_saver_hook = LayerOutputSaverHook()(model)
#         self.model = model

#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)

#     def validation_step(self, batch, batch_idx):
#         return self.model.validation_step(batch)

#     def test_step(self, batch, batch_idx):
#         return self.model.test_step(batch)

#     def training_step(self, *args: np.Any, **kwargs: np.Any):
#         return super().training_step(*args, **kwargs)


#     def teardown(self, stage):
#         for layer_name in self.layer_output_saver_hook.layers:
#             outputs = self.layer_output_saver_hook.output(layer_name)
#             out = outputs[0].detach().cpu().numpy()
#             out = out.reshape(out.shape[0], -1)

#             print(f"Creating a T-SNE for {layer_name}...")
#             transformed = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(out)

#             output_file = f"{layer_name}_transformed"
#             np.save(f"{output_file}.npy", transformed)
#             print(f"Saved transformed data to {output_file}")

#             fig = px.scatter(transformed, x=0, y=1)
#             fig.show()
#             fig.write_image(f"{output_file}.png")
#             print(f"Saved plot to {output_file}.png")


# class LatentAnalysis(ForwardHooker):
#     def __init__(
#         self,
#         model,
#         layers,
#         sklearn_technique_cls,
#         **sklearn_technique_kwargs,
#     ):
#         self.sklearn_models = {}
#         self.sklearn_creator = partial(
#             sklearn_technique_cls, **sklearn_technique_kwargs
#         )
#         super().__init__(model, layers)

#     def forward_hook(
#         self,
#         module,
#         inputs,
#         outputs,
#         layer_name: str,
#     ):
#         out = outputs.detach().cpu().numpy()
#         out = out.reshape(out.shape[0], -1)

#         if layer_name not in self.sklearn_models:
#             print(f"Creating a T-SNE for {layer_name}...")
#             self.sklearn_models[layer_name] = self.sklearn_creator()

#             print(f"Fit_transform T-SNE for {layer_name}...")
#             transformed = self.sklearn_models[layer_name].fit_transform(out)

#             output_file = f"{layer_name}_transformed"
#             np.save(f"{output_file}.npy", transformed)
#             print(f"Saved transformed data to {output_file}")

#             fig = px.scatter(transformed, x=0, y=1)
#             fig.show()
#             fig.write_image(f"{output_file}.png")
#             print(f"Saved plot to {output_file}.png")

#         else:
#             print(f"Using existing T-SNE for {layer_name}...")


class LatentAnalysis:
    def __init__(
        self,
        layers: List[str],
        sklearn_cls,
        output_name: str = "transformed",
        **sklearn_kwargs,
    ):
        self.layers = layers
        self.sklearn_class_constructor = partial(sklearn_cls, **sklearn_kwargs)
        self.output_name = output_name
        self._layer_outputs = defaultdict(list)
        self._handlers = dict()

    def forward_hook(
        self,
        module,
        inputs,
        outputs,
        layer_name: str,
    ):
        self._layer_outputs[layer_name].append(outputs.detach().cpu())

    def __call__(
        self,
        trainer: L.Trainer,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
    ):
        for layer_name in self.layers:
            torch_layer = attrgetter(layer_name)(model)
            handle = torch_layer.register_forward_hook(
                partial(self.forward_hook, layer_name=layer_name)
            )
            self._handlers[layer_name] = handle

        trainer.test(model, data_module)
        test_X, test_y = data_module.test_dataloader().dataset[:]
        test_y = np.stack(test_y)

        for layer_name, outputs in self._layer_outputs.items():
            outputs = torch.cat(outputs, dim=0)
            outputs = outputs.numpy()
            outputs = outputs.reshape(outputs.shape[0], -1)

            sklearn_model = self.sklearn_class_constructor()
            transformed_outputs = sklearn_model.fit_transform(outputs)

            output_file = f"{layer_name}_{self.output_name}"
            # np.save(f"{output_file}.npy", transformed_outputs)
            # print(f"Saved transformed data to {output_file}")

            fig = px.scatter(
                transformed_outputs,
                x=0,
                y=1,
                color=[str(i) for i in test_y],
                labels={"color": "Class Name"},
            )
            # fig.update_traces(marker=dict(color=test_y))
            # for i, name in enumerate(test_y):
            #     fig.data[i].name = str(name)
            fig.update_layout(showlegend=True)

            fig.write_image(f"{output_file}.png")
            print(f"Saved plot to {output_file}.png")

            handle = self._handlers[layer_name]
            handle.remove()

        return model


from dassl.modeling.ops.mixstyle import MixStyle, run_with_mixstyle, run_without_mixstyle

import torch
from sklearn.manifold import TSNE
from ssl_tools.data.data_modules.har import MultiModalHARSeriesDataModule
import lightning as L
from ssl_tools.models.nets.simple import SimpleClassificationNet


class Conv1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(6, 64, 5)
        self.mixstyle1 = MixStyle(0.7, 0.1, mix="random")
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout()

        self.conv2 = torch.nn.Conv1d(64, 64, 5)
        self.mixstyle2 = MixStyle(0.5, 0.1, mix="random")
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout()

        self.conv3 = torch.nn.Conv1d(64, 64, 5)
        self.mixstyle3 = MixStyle(0.2, 0.1, mix="random")
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.unsqueeze(x, 1)
        x = self.mixstyle1(x)
        x = torch.squeeze(x, 1)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = torch.unsqueeze(x, 1)
        x = self.mixstyle2(x)
        x = torch.squeeze(x, 1)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = torch.unsqueeze(x, 1)
        x = self.mixstyle3(x)
        x = torch.squeeze(x, 1)
        x = self.relu3(x)

        return x

class Simple1DConvNetwork(SimpleClassificationNet):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        """Create a simple 1D Convolutional Network with 3 layers and 2 fully
        connected layers.

        Parameters
        ----------
        input_shape : Tuple[int, int], optional
            A 2-tuple containing the number of input channels and the number of
            features, by default (6, 60).
        num_classes : int, optional
            Number of output classes, by default 6
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 1e-3
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_channels=input_shape[0])
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

    def _create_backbone(self, input_channels: int) -> torch.nn.Module:
        return Conv1D()
    
    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int]
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
            torch.nn.Dropout(),
            torch.nn.Linear(input_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, num_classes),
        )



model = Simple1DConvNetwork(input_shape=(6, 60), num_classes=7)
data_module = MultiModalHARSeriesDataModule(
    [
        "/workspaces/hiaac-m4/data/standartized_balanced/MotionSense",
        "/workspaces/hiaac-m4/data/standartized_balanced/RealWorld_thigh",
        "/workspaces/hiaac-m4/data/standartized_balanced/RealWorld_waist",
        "/workspaces/hiaac-m4/data/standartized_balanced/UCI",
        "/workspaces/hiaac-m4/data/standartized_balanced/WISDM",
    ],
    batch_size=64,
    num_workers=8,
)

test_data_module = MultiModalHARSeriesDataModule(
    "/workspaces/hiaac-m4/data/standartized_balanced/KuHar",
    batch_size=256,
    num_workers=8,
)

trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=20)],
    # limit_train_batches=10,
    # limit_val_batches=1,
)

analyser = LatentAnalysis(
    ["backbone.relu1", "backbone.relu2", "backbone.relu3", "fc.2"],
    TSNE,
    output_name="mix_style_transformed",
    n_components=2,
    random_state=42,
    perplexity=10,
)

print(model)

with run_without_mixstyle(model):
    trainer.fit(model, data_module)

analyser(trainer, model, test_data_module)

# print(model)


# trainer.test(model, data_module)


# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model_a = torch.nn.Sequential(
#             torch.nn.Linear(10, 10),
#             torch.nn.ReLU(),
#             torch.nn.Linear(10, 10),
#             torch.nn.ReLU(),
#             torch.nn.Linear(10, 10),
#             torch.nn.ReLU(),
#         )
#         self.model_b = torch.nn.Sequential(
#             torch.nn.Linear(10, 20),
#             torch.nn.ReLU(),
#             torch.nn.Linear(20, 10),
#             torch.nn.ReLU(),
#         )

#         self.model = torch.nn.Sequential(self.model_a, self.model_b)

#     def forward(self, x):
#         return self.model(x)


# model = Model()
# model = LatentAnalysis(
#     model,
#     ["model_a.0", "model_b.0", "model_b.3"],
#     sklearn_technique_cls=TSNE,
#     n_components=2,
#     random_state=42,
#     perplexity=5
# )

# model(torch.randn(10, 10))
