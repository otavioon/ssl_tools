from collections import defaultdict
from typing import List
import torch
from functools import partial
import lightning as L


from operator import attrgetter



class OutputLoggerCallback(L.Callback):
    def __init__(self, layers: List[str]):
        self.layers = layers
        self._layer_results = defaultdict(list)
        self._handlers = {}

    def count(self, module, input, output, layer_name: str):
        self._layer_results[layer_name].append(output.detach().cpu())

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ):
        if stage != "test":
            return

        for layer_name in self.layers:
            layer = attrgetter(layer_name)(pl_module)
            handle = layer.register_forward_hook(
                partial(self.count, layer_name=layer_name)
            )
            self._handlers[layer_name] = handle

    def teardown(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        for layer_name, handle in self._handlers.items():
            self._layer_results[layer_name] = torch.cat(
                self._layer_results[layer_name], dim=0
            )
            handle.remove()


# class LayerOutput:
#     def __init__(self, layer_name: str):
#         self.layer_name = layer_name
#         self.outputs = []

#     def __call__(self, module, input, output):
#         self.outputs.append(output.detach().cpu())


# class LayerOutputSaverHook:
#     def __init__(self, layers: List[str]) -> None:
#         self.layers = layers
#         self.layer_outputs = {layer: LayerOutput(layer) for layer in layers}

#     def output(self, layer_name: str):
#         return self.layer_outputs[layer_name].outputs

#     def __call__(self, module: torch.nn.Module | L.LightningModule):
#         for layer_name in self.layers:
#             layer = attrgetter(layer_name)(module)
#             layer.register_forward_hook(self.layer_outputs[layer_name])
#         return self


# class ForwardHooker(L.LightningModule):
#     def __init__(self, model, layers: List[str]):
#         super().__init__()
#         self.model = model
#         self.layers = layers

#         for layer_name in layers:
#             layer = attrgetter(layer_name)(model)
#             layer.register_forward_hook(
#                 partial(self.forward_hook, layer_name=layer_name)
#             )

#     def forward_hook(self, module, input, output, layer_name: str = ""):
#         # raise NotImplementedError("Implement this method in the subclass")
#         print(f"Layer: {layer_name}")


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


# def hook_fn(module, input, output):
#     print(type(module), type(input), type(output))
#     print(module, input[0].shape, output.shape)
#     print("---")


# model = Model()
# output_saver = LayerOutputSaverHook(["model_a.0", "model_b.0", "model_b.3"])(
#     model
# )

# # model = Model()
# # register_hook_in_layers(model, hook_fn, ['model_a.0', 'model_b.0'])

# model(torch.randn(10, 10))

# model(torch.randn(10, 10))

# model(torch.randn(10, 10))

# print(torch.cat(output_saver.output("model_a.0")).shape)
