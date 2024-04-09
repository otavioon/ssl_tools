from typing import Dict
import torch
import lightning as L


class SimpleClassificationNet(L.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        fc: torch.nn.Module,
        learning_rate: float = 1e-3,
        flatten: bool = True,
        loss_fn: torch.nn.Module = None,
        train_metrics: Dict[str, torch.Tensor] = None,
        val_metrics: Dict[str, torch.Tensor] = None,
        test_metrics: Dict[str, torch.Tensor] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.fc = fc
        self.learning_rate = learning_rate
        self.flatten = flatten
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.metrics = {
            "train": train_metrics or {},
            "val": val_metrics or {},
            "test": test_metrics or {},
        }

    def loss_func(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)
        return loss

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        if self.flatten:
            x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def compute_metrics(self, y_hat, y, step_name):
        for metric_name, metric_fn in self.metrics[step_name].items():
            metric = metric_fn.to(self.device)(y_hat, y)
            self.log(
                f"{step_name}_{metric_name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def single_step(self, batch: torch.Tensor, batch_idx: int, step_name: str):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.compute_metrics(y_hat, y, step_name)
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer


class SimpleReconstructionNet(L.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        learning_rate: float = 1e-3,
        loss_fn: torch.nn.Module = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn or torch.nn.MSELoss()

    def loss_func(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)
        return loss

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        return x

    def single_step(self, batch: torch.Tensor, batch_idx: int, step_name: str):
        x, _ = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, x)
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer


class MLPClassifier(SimpleClassificationNet):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
        learning_rate: float = 1e-3,
        flatten: bool = True,
        loss_fn: torch.nn.Module = None,
        train_metrics: Dict[str, torch.Tensor] = None,
        val_metrics: Dict[str, torch.Tensor] = None,
        test_metrics: Dict[str, torch.Tensor] = None,
    ):
        self.hidden_size = hidden_size
        backbone = torch.nn.Sequential()
        for i in range(num_hidden_layers):
            if i == 0:
                backbone.add_module(
                    f"fc{i+1}", torch.nn.Linear(input_size, hidden_size)
                )
            else:
                backbone.add_module(
                    f"fc{i+1}", torch.nn.Linear(hidden_size, hidden_size)
                )
            backbone.add_module(f"relu{i+1}", torch.nn.ReLU())
        fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Softmax(dim=1),
        )
        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            flatten=flatten,
            loss_fn=loss_fn,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )


# def main():
#     from torch.utils.data import DataLoader

#     class RandomDataset:
#         def __init__(
#             self,
#             num_samples: int = 64,
#             num_classes: int = 6,
#             input_shape: tuple = (6, 60),
#         ):
#             self.num_samples = num_samples
#             self.num_classes = num_classes
#             self.input_shape = input_shape

#         def __len__(self):
#             return self.num_samples

#         def __getitem__(self, idx):
#             return (
#                 torch.randn(*self.input_shape),
#                 torch.randint(0, self.num_classes, (1,)).item(),
#             )

#     class RandomDataModule(L.LightningDataModule):
#         def __init__(
#             self, num_samples, num_classes, input_shape, batch_size: int = 1
#         ):
#             super().__init__()
#             self.num_samples = num_samples
#             self.num_classes = num_classes
#             self.input_shape = input_shape
#             self.batch_size = batch_size

#         def train_dataloader(self):
#             return DataLoader(
#                 RandomDataset(
#                     self.num_samples, self.num_classes, self.input_shape
#                 ),
#                 batch_size=self.batch_size,
#             )

#     data_module = RandomDataModule(
#         num_samples=10, num_classes=6, input_shape=(6 * 60,), batch_size=8
#     )

#     model = MLPClassificator(
#         input_size=6 * 60, hidden_size=128, num_hidden_layers=2, output_size=6
#     )

#     trainer = L.Trainer(
#         max_epochs=1, logger=False, devices=1, accelerator="gpu"
#     )

#     trainer.fit(model, datamodule=data_module)


# if __name__ == "__main__":
#     main()
