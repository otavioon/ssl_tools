from typing import Dict
import torch
import lightning as L
from torchmetrics import Accuracy


class Simple1DConvNetwork(L.LightningModule):
    """Model for human-activity-recognition."""

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 6,
        time_steps: int = 60,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.learning_rate = learning_rate
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.metrics = {
            "acc": Accuracy(task="multiclass", num_classes=num_classes)
        }

        # Extract features, 1D conv layers
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 64, 5),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv1d(64, 64, 5),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv1d(64, 64, 5),
            torch.nn.ReLU(),
        )
        
        self.fc_input_features = self._calculate_fc_input_features(input_channels)
        
        # Classify output, fully connected layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(self.fc_input_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, num_classes),
        )

    def _calculate_fc_input_features(self, input_channels):
        # Dummy input to get the output shape after conv2
        x = torch.randn(1, input_channels, self.time_steps)
        with torch.no_grad():
            out = self.features(x)
        # Return the total number of features
        return out.view(out.size(0), -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.fc_input_features)
        out = self.classifier(x)
        return out
    
    def loss_function(self, X, y):
        loss = self.loss_func(X, y)
        return loss

    def _common_step(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log(
            f"{prefix}_loss",
            loss,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        return loss, y_hat, y

    def _compute_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, stage: str
    ) -> Dict[str, float]:
        """Compute the metrics.

        Parameters
        ----------
        y_hat : torch.Tensor
            The predictions of the model
        y : _type_
            The ground truth labels
        stage : str
            The stage of the training loop (train, val or test)

        Returns
        -------
        Dict[str, float]
            A dictionary containing the metrics. The keys are the names of the
            metrics, and the values are the values of the metrics.
        """
        return {
            f"{stage}_{metric_name}": metric.to(self.device)(y_hat, y)
            for metric_name, metric in self.metrics.items()
        }

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, prefix="val")

        if self.metrics is not None:
            results = self._compute_metrics(y_hat, y, "val")
            self.log_dict(
                results,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, prefix="test")

        if self.metrics is not None:
            results = self._compute_metrics(y_hat, y, "test")
            self.log_dict(
                results,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer



# FROM https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class Simple2DConvNetwork(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 10,
        num_classes: int = 6,
        time_steps: int = 60,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.metrics = {
            "acc": Accuracy(task="multiclass", num_classes=num_classes)
        }

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=(1, input_channels),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, input_channels)
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )

        self.fc_input_features = self._calculate_fc_input_features(
            input_channels
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.fc_input_features, out_features=1000
            ),
            torch.nn.ReLU(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1000, out_features=500), torch.nn.ReLU()
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=500, out_features=num_classes)
        )

    def _calculate_fc_input_features(self, input_channels):
        # Dummy input to get the output shape after conv2
        x = torch.randn(1, input_channels, 1, self.time_steps)
        with torch.no_grad():
            out = self.conv1(x)
            out = self.conv2(out)
        # Return the total number of features
        return out.view(out.size(0), -1).size(1)

    def loss_function(self, X, y):
        loss = self.loss_func(X, y)
        return loss

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # Flatten the output for fully
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def _common_step(self, batch, batch_idx, prefix):
        x, y = batch
        if x.ndim == 3:
            x = x.unsqueeze(2)
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log(
            f"{prefix}_loss",
            loss,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        return loss, y_hat, y

    def _compute_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, stage: str
    ) -> Dict[str, float]:
        """Compute the metrics.

        Parameters
        ----------
        y_hat : torch.Tensor
            The predictions of the model
        y : _type_
            The ground truth labels
        stage : str
            The stage of the training loop (train, val or test)

        Returns
        -------
        Dict[str, float]
            A dictionary containing the metrics. The keys are the names of the
            metrics, and the values are the values of the metrics.
        """
        return {
            f"{stage}_{metric_name}": metric.to(self.device)(y_hat, y)
            for metric_name, metric in self.metrics.items()
        }

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, prefix="val")

        if self.metrics is not None:
            results = self._compute_metrics(y_hat, y, "val")
            self.log_dict(
                results,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, prefix="test")

        if self.metrics is not None:
            results = self._compute_metrics(y_hat, y, "test")
            self.log_dict(
                results,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
