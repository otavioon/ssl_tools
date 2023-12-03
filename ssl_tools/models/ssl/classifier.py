from typing import Any, Dict
import lightning as L
import torch
from torchmetrics import Metric


class SSLDiscriminator(L.LightningModule):
    def __init__(
        self,
        backbone,
        head,
        loss_fn,
        learning_rate: float = 1e-3,
        update_backbone: bool = True,
        metrics: Dict[str, Metric] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.update_backbone = update_backbone
        self.metrics = metrics

    def _loss_func(self, y_hat, y):
        return self.loss_fn(y_hat, y)
    
    def _compute_metrics(self, y_hat, y, stage: str):
        return {
            f"{stage}_{metric_name}": metric.to(self.device)(y_hat, y)
            for metric_name, metric in self.metrics.items()
        }
            
            

    def forward(self, x):
        encodings = self.backbone.forward(x)
        predictions = self.head.forward(encodings)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self._loss_func(predictions, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self._loss_func(predictions, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if self.metrics is not None:
            results = self._compute_metrics(predictions, y, "val")
            self.log_dict(
                results,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self._loss_func(predictions, y)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if self.metrics is not None:
            for metric in self.metrics:
                metric(predictions, y)
                self.log(
                    f"train_{metric.name}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
        return loss

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch
        predictions = self.forwardf(x)
        return predictions

    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        if self.update_backbone:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self._freeze(self.backbone)
            return torch.optim.Adam(
                self.head.parameters(), lr=self.learning_rate
            )
