from dataclasses import dataclass
from datetime import datetime
from operator import attrgetter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from torchmetrics import Accuracy
from functools import partial
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from collections import defaultdict
from contextlib import contextmanager
import pandas as pd
import torch


class LayerOutputSaverHook:
    def __init__(self):
        self._layer_outputs = defaultdict(list)
        self._handlers = {}

    def _forward_hook(
        self,
        module,
        inputs,
        outputs,
        layer_name: str,
    ):
        # Save the outputs of the selected layers
        self._layer_outputs[layer_name].append(outputs.detach().cpu())

    @contextmanager
    def run_model_with_hooks(
        self, model: L.LightningModule, layer_names: List[str]
    ):
        self.attach_hooks(model, layer_names)
        yield
        self.remove_hooks()

    def attach_hooks(self, model: L.LightningModule, layer_names: List[str]):
        for layer_name in layer_names:
            torch_layer = attrgetter(layer_name)(model)
            handle = torch_layer.register_forward_hook(
                partial(self._forward_hook, layer_name=layer_name)
            )
            self._handlers[layer_name] = handle

    def remove_hooks(self):
        for handle in self._handlers.values():
            handle.remove()

    def outputs_from_layer(self, layer_name: str, concat: bool = True):
        outputs = self._layer_outputs[layer_name]
        if concat:
            outputs = torch.cat(outputs, dim=0)
        return outputs


class LatentAnalysis:
    def __init__(
        self,
        layers: List[str],
        sklearn_cls,
        output_name_suffix: str = "transformed",
        **sklearn_kwargs,
    ):
        self.layers = layers
        self.layer_outputs = LayerOutputSaverHook()
        self.sklearn_class_constructor = partial(sklearn_cls, **sklearn_kwargs)
        self.output_name_suffix = output_name_suffix

    def __call__(
        self,
        trainer: L.Trainer,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
    ):
        dfs = {}

        with self.layer_outputs.run_model_with_hooks(model, self.layers):
            # Do a test pass to get the outputs of the selected layers
            y_hat = trainer.predict(model, data_module)
            y_hat = torch.cat(y_hat, dim=0)
            y_hat = torch.argmax(y_hat, dim=1)
            y_hat = y_hat.numpy()

            # Obtain the true labels
            # data_module.setup("test")
            X, y = data_module.predict_dataloader().dataset[:]
            y = np.stack(y)

            # For each layer, apply the sklearn technique
            for layer_name in self.layers:
                outputs = self.layer_outputs.outputs_from_layer(
                    layer_name, concat=True
                )
                outputs = outputs.numpy()
                outputs = outputs.reshape(outputs.shape[0], -1)

                # Create a sklearn model and fit_transform the data
                sklearn_model = self.sklearn_class_constructor()
                transformed_outputs = sklearn_model.fit_transform(outputs)

                df = pd.DataFrame(transformed_outputs)
                df["label"] = y
                df["predicted"] = y_hat
                df.to_csv(
                    f"{layer_name}_{self.output_name_suffix}.csv", index=False
                )
                print(
                    f"Saved transformed data to {layer_name}_{self.output_name_suffix}.csv"
                )
                dfs[layer_name] = df

        return dfs
