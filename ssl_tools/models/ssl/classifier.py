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
        """A generic SSL Discriminator model. It takes a backbone and a head
        and trains them jointly (or not, depending on the ``update_backbone``
        parameter).

        In summary, the training loop is as follows:
        1. Forward pass through the backbone
        2. Forward pass through the head (with the backbone's output)
        3. Compute the loss
        4. Backpropagate the loss through the head and the backbone (the latter
        is backpropagated only if ``update_backbone`` is True)

        Parameters
        ----------
        backbone : _type_
            The backbone of the model, that will encode the input data
        head : _type_
            The head of the model, that will make the final predictions. It
            taks the output of the backbone as input.
        loss_fn : _type_
            The loss function to use. By default, it is a cross entropy loss.
        learning_rate : float, optional
            The learning rate to use for the optimizer.
        update_backbone : bool, optional
            If True, the backbone will be updated during training. Otherwise,
            only the head will be updated.
        metrics : Dict[str, Metric], optional
            The metrics to use during training. The keys are the names of the
            metrics, and the values are the metrics themselves.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.update_backbone = update_backbone
        self.metrics = metrics

    def _loss_func(self, y_hat: torch.Tensor, y: torch.Tensor):
        """Calculates the loss function.

        Parameters
        ----------
        y_hat : torch.Tensor
            The predictions of the model
        y : torch.Tensor
            The ground truth labels
        """
        return self.loss_fn(y_hat, y)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the model. It first passes the input
        through the backbone, and then passes the output of the backbone through
        the head.

        Parameters
        ----------
        x : torch.Tensor
            The input data. If it is a tuple or a list, it will be unpacked
            before being passed to the backbone.

        Returns
        -------
        torch.Tensor
            The predictions of the model.
        """
        if isinstance(x, tuple) or isinstance(x, list):
            encodings = self.backbone.forward(*x)
        else:
            encodings = self.backbone.forward(x)
        predictions = self.head.forward(encodings)
        return predictions

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """Performs a training step. It first performs a forward pass through
        the model, then computes the loss. Finally, it logs the loss
        (train_loss)

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data
        batch_idx : int
            The index of the batch

        Returns
        -------
        torch.Tensor
            The loss of the model
        """
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

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """Performs a validation step. It first performs a forward pass through
        the model, then computes the loss. Finally, it logs the loss and the
        metrics (if any) (val_loss, val_metric_1, val_metric_2, ...)

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data
        batch_idx : int
            The index of the batch

        Returns
        -------
        torch.Tensor
            The loss of the model
        """
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

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs a test step. It first performs a forward pass through
        the model, then computes the loss. Finally, it logs the loss and the
        metrics (if any) (test_loss, test_metric_1, test_metric_2, ...)

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data
        batch_idx : int
            The index of the batch

        Returns
        -------
        torch.Tensor
            The loss of the model
        """
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
            results = self._compute_metrics(predictions, y, "test")
            self.log_dict(
                results,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs a prediction step. It only performs a forward pass through
        the model.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data
        batch_idx : int
            The index of the batch

        Returns
        -------
        torch.Tensor
            The predictions of the model
        """
        x, y = batch
        predictions = self.forwardf(x)
        return predictions

    def _freeze(self, model):
        """Freezes the model, i.e. sets the requires_grad parameter of all the
        parameters to False.

        Parameters
        ----------
        model : _type_
            The model to freeze
        """
        for param in model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        """Configures the optimizer. If ``update_backbone`` is True, it will
        update the parameters of the backbone and the head. Otherwise, it will
        only update the parameters of the head.
        """
        if self.update_backbone:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self._freeze(self.backbone)
            return torch.optim.Adam(
                self.head.parameters(), lr=self.learning_rate
            )
