import torch
import lightning as L
import numpy as np
from typing import Any

from torchmetrics.functional import accuracy
from ssl_tools.utils.configurable import Configurable


class CPC(L.LightningModule, Configurable):
    """Implements the Contrastive Predictive Coding (CPC) model, as described in
    https://arxiv.org/abs/1807.03748. The implementation was adapted from
    https://github.com/sanatonek/TNC_representation_learning

    The model is trained to predict the future representation of a window of
    samples, given the past representation of the same window. The model is
    trained to maximize the mutual information between the past and future
    representations. The model is trained using a contrastive loss, where the
    positive samples are the future representations of the same window, and the
    negative samples are the future representations of other windows.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        density_estimator: torch.nn.Module,
        auto_regressor: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        window_size: int = 4,
        n_size: int = 5,
    ):
        """Implements the Contrastive Predictive Coding (CPC) model.

        Parameters
        ----------
        encoder : torch.nn.Module
            Model that encodes a window of samples into a representation.
            This model takes the sample windows as input and outputs a
            representation of the windows (Z-vector). Original paper uses a
            GRU+Linear.
        density_estimator : torch.nn.Module
            Model that estimates the density of a representation.
        auto_regressor : torch.nn.Module
            Model that predicts the future representation of a window of
            samples, given the past representation of the same window.
        lr : float, optional
            Learning rate, by default 1e-3.
        weight_decay : float, optional
            Weight decay, by default 0.0
        window_size : int, optional
            Size of the window which will be used to generate the Z-vector.
            Samples are divided into windows of size window_size and than
            encoded into a Z-vector. The Z-vector is used to train the
            density_estimator and auto_regressor models. The Z-vector is also
            used to train the encoder model, but the encoder model is trained
            to maximize the mutual information between the Z-vector and the
            future Z-vector, not to minimize the density estimation error.
            By default 4
        """
        super().__init__()
        self.encoder = encoder
        self.density_estimator = density_estimator
        self.auto_regressor = auto_regressor
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.n_size = n_size
        self.loss_func = torch.nn.CrossEntropyLoss()

    def loss_function(self, X_N, labels):
        loss = self.loss_func(X_N.view(1, -1), labels)
        return loss

    def forward(self, sample):
        # Remove dimension of size 1, if present
        sample = sample.squeeze(0)

        # Select a random timestamp to subsample the sample in order to not
        # use the full sample. The random timestamp is an index in the range
        # [5 * window_size,  sample_size - 5 * window_size]. Thus, we ensure
        # that the random timestamp is not too close to the beginning or end
        # of the sample.
        random_timestamp = np.random.randint(
            5 * self.window_size, sample.shape[-1] - 5 * self.window_size
        )

        # Resample the sample. We remove timestamps 20 * window_size before and
        # after the random timestamp. The sample will have a new length of
        # 40 * window_size (20 * window_size before and after the random), or
        # less if the sample is too short or the random timestamp is too close
        # to the beginning or end of the sample.
        sample = sample[
            :,
            max(0, (random_timestamp - 20 * self.window_size)) : min(
                sample.shape[-1], random_timestamp + 20 * self.window_size
            ),
        ]
        # Convert to tensor and move to device (cpu)
        sample = torch.tensor(sample).cpu()

        # Get the length of the sample
        sample_length = sample.shape[-1]

        # Split the array into windows of size window_size. These windows will be
        # used to generate the Z-vector. The last window is discarded if it is
        # not complete.
        windowed_sample = np.split(
            ary=sample[
                :, : (sample_length // self.window_size) * self.window_size
            ],
            indices_or_sections=(sample_length // self.window_size),
            axis=-1,
        )
        # As result is a list of windows, we stack them to get a tensor and move
        # to desired device
        windowed_sample = torch.tensor(
            np.stack(windowed_sample, 0), device=self.device
        )
        # Encode the windows into a Z-vector.
        encodings = self.encoder(windowed_sample)

        # Given the z-vector of windows, we select a random window to be our
        # random timestamp. Elements before and after the random timestamp are
        # considered "past" and "future" respectively. We use the "past" to
        # train the encoder model to maximize the mutual information between
        # the "past" and "future" representations. We use the "future" to train
        # the density_estimator and auto_regressor models.
        # We select a random window in the range [2, len(encodings) - 2].
        # Thus, we ensure that "past" and "future" have at least 2 elements.
        window_ind = torch.randint(2, len(encodings) - 2, size=(1,))

        # Generate the context vector c_t using the "past" representations.
        _, c_t = self.auto_regressor(
            encodings[max(0, window_ind[0] - 10) : window_ind[0] + 1].unsqueeze(
                0
            )
        )
        densities = self.density_estimator(c_t.squeeze(1).squeeze(0))
        density_ratios = torch.bmm(
            encodings.unsqueeze(1),
            densities.expand_as(encodings).unsqueeze(-1),
        )
        density_ratios = density_ratios.view(
            -1,
        )
        r = set(range(0, window_ind[0] - 2))
        r.update(set(range(window_ind[0] + 3, len(encodings))))
        rnd_n = np.random.choice(list(r), self.n_size)
        # Generate the encoded representations
        X_N = torch.cat(
            [
                density_ratios[rnd_n],
                density_ratios[window_ind[0] + 1].unsqueeze(0),
            ],
            0,
        )
        return X_N

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "val")
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "test")
        return loss
    
    def predict_step(self, batch, batch_idx):
        return self.forward(batch)
    
    def _shared_step(self, batch, batch_idx, prefix):
        assert len(batch) == 1, "Batch must be 1 sample only"
        assert batch.shape[-1] > 5 * self.window_size, "Sample too short"

        # Generate the encoded representations
        X_N = self.forward(batch)
        # Generate the labels
        labels = torch.Tensor([len(X_N) - 1]).to(self.device).long()
        # Calculate the loss
        loss = self.loss_function(X_N, labels)
        # Log the loss
        self.log(
            f"{prefix}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        # learnable_parameters = (
        #     list(self.density_estimator.parameters())
        #     + list(self.encoder.parameters())
        #     + list(self.auto_regressor.parameters())
        # )
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def get_config(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "window_size": self.window_size,
        }


# class CPC_Classifier(L.LightningModule):
#     def __init__(
#         self,
#         encoder: torch.nn.Module,
#         classifier: torch.nn.Module,
#         lr: float = 1e-3,
#         weight_decay: float = 0.0,
#         task_class: str = "multiclass",
#         num_classes: int = 6,
#     ):
#         super().__init__()
#         self.encoder = encoder.to(self.device)
#         self.classifier = classifier.to(self.device)
#         self.learning_rate = lr
#         self.weight_decay = weight_decay
#         self.training_step_losses = []
#         self.validation_step_losses = []
#         self.loss_function = torch.nn.CrossEntropyLoss()
#         self.task_class = task_class
#         self.num_classes = num_classes

#     def configure_optimizers(self) -> Any:
#         optimizer = torch.optim.Adam(
#             self.classifier.parameters(),
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay,
#         )
#         return optimizer

#     def forward(self, x):
#         encodings = self.encoder(x)
#         predictions = self.classifier(encodings)
#         return predictions

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         predictions = self.forward(x)
#         loss = self.loss_function(predictions, y.long())
#         self.training_step_losses.append(loss)
#         return loss

#     def on_train_epoch_end(self) -> None:
#         # do something with all training_step outputs, for example:
#         epoch_mean = torch.stack(self.training_step_losses).mean()
#         self.log(
#             "train_loss",
#             epoch_mean,
#             on_epoch=True,
#             on_step=False,
#             prog_bar=True,
#             logger=True,
#         )
#         # free up the memory
#         self.training_step_losses.clear()

#     def validation_step(self, batch, batch_idx):
#         loss, acc = self._shared_eval_step(batch, batch_idx)
#         self.validation_step_losses.append(loss)
#         metrics = {"val_acc": acc, "val_loss": loss}
#         self.log_dict(metrics)
#         return metrics

#     def test_step(self, batch, batch_idx):
#         loss, acc = self._shared_eval_step(batch, batch_idx)
#         metrics = {"test_acc": acc, "test_loss": loss}
#         self.log_dict(metrics)
#         return metrics

#     def on_validation_epoch_end(self) -> None:
#         # do something with all training_step outputs, for example:
#         epoch_mean = torch.stack(self.validation_step_losses).mean()
#         self.log(
#             "val_loss",
#             epoch_mean,
#             on_epoch=True,
#             on_step=False,
#             prog_bar=True,
#             logger=True,
#         )
#         # free up the memory
#         self.validation_step_losses.clear()

#     def _shared_eval_step(self, batch, batch_idx):
#         x, y = batch
#         predictions = self.forward(x)
#         loss = self.loss_function(predictions, y.long())
#         acc = accuracy(
#             torch.argmax(predictions, dim=1),
#             y.long(),
#             task=self.task_class,
#             num_classes=self.num_classes,
#         )
#         return loss, acc
