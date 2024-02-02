from typing import Tuple
import torch
import lightning as L
import numpy as np

from ssl_tools.utils.configurable import Configurable
from ssl_tools.models.layers.gru import GRUEncoder


class TNCDiscriminator(torch.nn.Module):
    def __init__(self, input_size: int = 10, n_classes: int = 1):
        """Simple discriminator network. As usued by `Tonekaboni et al.`
        at "Unsupervised Representation Learning for Time Series with Temporal
        Neighborhood Coding" (https://arxiv.org/abs/2106.00750)

        It is composed by:
            - Linear(2 * ``input_size``, 4 * ``input_size``)
            - ReLU
            - Dropout(0.5)
            - Linear(4 * ``input_size``, ``n_classes``)
        Parameters
        ----------
        input_size : int, optional
            Size of the input sample, by default 10
        n_classes : int, optional
            Number of output classes (output_size), by default 1
        """
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # Defines the model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2 * self.input_size, 4 * self.input_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4 * self.input_size, self.n_classes),
        )
        # Init the weights of linear layers with xavier uniform method
        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x):
        """
        Predict the probability of the two inputs belonging to the same
        neighbourhood.
        """
        return self.model(x)


class TNC(L.LightningModule, Configurable):
    def __init__(
        self,
        discriminator: torch.nn.Module,
        encoder: torch.nn.Module,
        mc_sample_size: int = 20,
        w: float = 0.05,
        learning_rate=1e-3,
    ):
        """Implements the Temporal Neighbourhood Contrastive (TNC) model, as
        described in https://arxiv.org/pdf/2106.00750.pdf. The implementation
        was adapted from https://github.com/sanatonek/TNC_representation_learning

        Parameters
        ----------
        discriminator : torch.nn.Module
            A discriminator model that takes as input the concatenation of the
            representation of the current time step and the representation of
            the positive/negative samples. It is a binary classifier that
            predict if the samples are neighbors or not.
        encoder : torch.nn.Module
            Encode a window of samples into a representation. This model is
            usually a GRU that encodes the samples into a representation of
            fixed encoding size.
        mc_sample_size : int
            The number of close and distant samples selected in the dataset.
        w : float
            This parameter is used in loss and represent probability of
            sampling a positive window from the non-neighboring region.
        learning_rate : _type_, optional
            The learning rate of the optimizer, by default 1e-3
        """
        super().__init__()
        self.discriminator = discriminator
        self.encoder = encoder
        self.mc_sample_size = mc_sample_size
        self.w = w
        self.learning_rate = learning_rate
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def loss_function(self, y: torch.Tensor, y_hat: torch.Tensor):
        """Calculate the loss.

        Parameters
        ----------
        y : torch.Tensor
            The ground truth labels.
        y_hat : torch.Tensor
            The predicted labels.
        """
        return self.loss_func(y, y_hat)

    def forward(self, x):
        return self.encoder(x)

    def _shared_step(
        self,
        x_t: torch.Tensor,
        x_p: torch.Tensor,
        x_n: torch.Tensor,
        stage: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs TNC and returns the representation and the loss.

        Parameters
        ----------
        x_t : torch.Tensor
            A tensor with the sample of the current time step. It is expected
            to be the shape (B, C, T), where B is the batch size, C is the
            number of channels (features) and T is the number of time steps.
        x_p : torch.Tensor
            A set of positive samples. It is expected to be the shape
            (B * mc_sample_size, C, T), where B is the batch size, C is the
            number of channels (features) and T is the number of time steps.
        x_n : torch.Tensor
            A set of negative samples. It is expected to be the shape
            (B * mc_sample_size, C, T), where B is the batch size, C is the
            number of channels (features) and T is the number of time steps.
        stage : str
            Stage of the training (train, val, test)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A 2-element tuple containing the representation and the loss,
            respectively.
        """
        batch_size, f_size, len_size = x_t.shape

        # Ensures x_p and x_n to be of shape
        # (batch_size * mc_sample_size, f_size, len_size)
        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))

        # X_t is a single sample, so we need to repeat it mc_sample_size times
        # to match the shape of x_p and x_n
        x_t = torch.repeat_interleave(x_t, self.mc_sample_size, axis=0)

        # Vectors with neighbors (1s) and non-neighbors labels (0s)
        neighbors = torch.ones((len(x_p)), device=self.device)
        non_neighbors = torch.zeros((len(x_n)), device=self.device)

        # Encoding features (usually using a GRU encoding) that will return a
        # representation of shape (batch_size, encoding_size)
        z_t = self.forward(x_t)
        z_p = self.forward(x_p)
        z_n = self.forward(x_n)

        # Discriminate features. The discriminator is usually an MLP that,
        # performs a binary classification of the samples (return 0s and 1s).
        # In fact, we try to discriminate if the samples are neighbors or not.
        # Positive samples (1/2 original, 1/2 positive ones)
        z_tp = torch.cat([z_t, z_p], -1)
        d_p = self.discriminator(z_tp).view((-1, ))
        # Negative samples (1/2 original, 1/2 negative ones)
        z_tn = torch.cat([z_t, z_n], -1)
        d_n = self.discriminator(z_tn).view((-1, ))

        # Compute loss positive loss (positive pairs vs. neighbours)
        p_loss = self.loss_function(d_p, neighbors)
        # Compute loss negative loss (negative pairs vs. non_neighbors)
        n_loss = self.loss_function(d_n, non_neighbors)
        # Compute loss of negative pairs vs. neighbours (probability of
        # sampling a positive window from the non-neighboring region)
        n_loss_u = self.loss_function(d_n, neighbors)
        # Compute the final loss
        loss = (p_loss + self.w * n_loss_u + (1 - self.w) * n_loss) / 2

        # Log the loss
        self.log(
            f"{stage}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        return (z_t, loss)

    def training_step(self, batch, batch_idx):
        x_t, x_p, x_n = batch
        _, loss = self._shared_step(x_t, x_p, x_n, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x_t, x_p, x_n = batch
        _, loss = self._shared_step(x_t, x_p, x_n, "val")
        return loss

    def test_step(self, batch, batch_idx):
        x_t, x_p, x_n = batch
        _, loss = self._shared_step(x_t, x_p, x_n, "test")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_config(self) -> dict:
        return {
            "mc_sample_size": self.mc_sample_size,
            "w": self.w,
            "learning_rate": self.learning_rate,
        }


def build_tnc(
    encoding_size: int = 150,
    in_channel: int = 6,
    mc_sample_size: int = 20,
    w: float = 0.05,
    learning_rate=1e-3,
    gru_hidden_size: int = 100,
    gru_num_layers: int = 1,
    gru_bidirectional: bool = True,
    dropout: float = 0.0,
) -> TNC:
    """Builds a TNC model. This function aids the creation of a TNC model
    by providing default values for the parameters. 

    Parameters
    ----------
    encoding_size : int, optional
        The size of the encoding. This is the size of the representation.
    in_channel : int, optional
        The number of channels (features) of the input samples (e.g., 6 for
        the MotionSense dataset)
    mc_sample_size : int
        The number of close and distant samples selected in the dataset.
    w : float
        This parameter is used in loss and represent probability of
        sampling a positive window from the non-neighboring region.
    learning_rate : _type_, optional
        The learning rate of the optimizer
    gru_hidden_size : int, optional
        The number of features in the hidden state of the GRU.
    gru_num_layers : int, optional
        Number of recurrent layers in the GRU. E.g., setting ``num_layers=2``
        would mean stacking two GRUs together to form a `stacked GRU`,
        with the second GRU taking in outputs of the first GRU and
        computing the final results.
    gru_bidirectional : bool, optional
        If ``True``, becomes a bidirectional GRU.
    dropout : float, optional
        The dropout probability.

    Returns
    -------
    TNC
        The TNC model.
    """
    discriminator = TNCDiscriminator(input_size=encoding_size, n_classes=1)

    encoder = GRUEncoder(
        hidden_size=gru_hidden_size,
        in_channels=in_channel,
        encoding_size=encoding_size,
        num_layers=gru_num_layers,
        dropout=dropout,
        bidirectional=gru_bidirectional,
    )

    model = TNC(
        discriminator=discriminator,
        encoder=encoder,
        mc_sample_size=mc_sample_size,
        w=w,
        learning_rate=learning_rate,
    )

    return model
