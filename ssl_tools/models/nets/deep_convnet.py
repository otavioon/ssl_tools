from typing import Dict
import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy


class DeepConvNet(L.LightningModule):

    def __init__(
        self,
        input_channels: int = 6,
        time_steps: int = 60,
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.time_steps = time_steps
        self.output_channels = num_classes
        self.learning_rate = learning_rate
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.metrics = {
            "acc": Accuracy(task="multiclass", num_classes=num_classes)
        }

        self.features = nn.Sequential(
            nn.Conv2d(
                self.input_channels, 64, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # nn.Dropout(p=0.5)
        )

        self.fc_input_features = self._calculate_fc_input_features(
            input_channels, time_steps
        )

        self.classifier = nn.Sequential(
            nn.Linear(4, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def _calculate_fc_input_features(self, input_channels: int, time_steps: int) -> int:
        """Calculate the number of input features of the fully connected layer.
        Basically, it performs a forward pass with a dummy input to get the
        output shape after the convolutional layers.

        Parameters
        ----------
        input_channels : int
            The number of input channels.

        Returns
        -------
        int
            The number of input features of the fully connected layer.
        """

        # Dummy input to get the output shape after conv2
        x = torch.randn(1, input_channels, time_steps, 1)
        with torch.no_grad():
            out = self.features(x)
        # Return the total number of features
        return out.view(out.size(0), -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.fc_input_features)
        x = self.classifier(x)
        return x
    
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
        y : torch.Tensor
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
    
    
def main():
    print("OLA")
    model = DeepConvNet()
    print(model)
    print(model.fc_input_features)
    # x = torch.randn(16, 60, 60)
    # y = model(x)
    # print(y.size())
    
if __name__ == "__main__":
    main()