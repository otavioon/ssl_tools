import torch
import lightning as L
import numpy as np

from ssl_tools.utils.configurable import Configurable
from ssl_tools.models.layers.gru import GRUEncoder
from .tnc import TNCHead


class CPCHead(TNCHead):
    pass


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
            Size of the input windows (X_t) to be fed to the encoder
        n_size : int, optional
            Number of negative samples to be used in the contrastive loss
            (steps to predict)
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
        loss = self.loss_func(X_N, labels)
        return loss

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model. This method is used to
        encode a sample into a representation.

        Parameters
        ----------
        sample : torch.Tensor
            The sample to be encoded. A tensor of shape (B, C, T), where B is
            the batch size, C is the number of channels and T is the number of
            time steps.

        Returns
        -------
        torch.Tensor
            A tensor of size (B, encoder_output_size), with the samples encoded.
        """
        return self.encoder(sample)

    def _step(self, sample: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Parameters
        ----------
        sample : torch.Tensor
            A tensor of shape (B, C, T), where B is the batch size, C is the
            number of channels and T is the number of time steps. Note that T
            may vary between samples of the dataset. If this is the case, the
            B dimension must be 1.

        Returns
        -------
        torch.Tensor
            A tensor of size (B, encoder_output_size), with the samples encoded
        """
        # ----------------------------------------------------------------------
        # 1. Split the sample X into windows of size window_size (X_t)
        # ----------------------------------------------------------------------

        # Remove the batch dimension if it is 1, and get the sample size (T)
        sample = sample.squeeze(0)
        time_len = sample.shape[-1]

        # Select a random time step in the range
        # [5 * window_size, T - 5 * window_size]
        # Just to make sure we have enough samples before and after the random
        # time step
        random_centering_t = np.random.randint(
            5 * self.window_size, time_len - 5 * self.window_size
        )

        # Here we center the sample around the random timestamp, and only keep
        # 40 * window_size time steps around it (20 before and 20 after)
        # +-------------------------------------------------------------------+
        # | 20 * window_size ...  random_centering_t  ... 20 * window_size    |
        # +-------------------------------------------------------------------+
        sample = sample[
            :,  # The channels are not affected
            max(0, (random_centering_t - 20 * self.window_size)) : min(
                sample.shape[-1], random_centering_t + 20 * self.window_size
            ),
        ]
        # Update the time_len (40 * window_size or less), and move sample to CPU
        time_len = sample.shape[-1]
        sample = sample.cpu()

        # Split the sample into windows of size ``window_size``. This generates
        # the inputs (X_t, X_{t+1}, ..., X_{t+window_size-1}), which is a list
        # of 40 tensors of shape (C, window_size)
        X_ts = np.split(
            # Crop the end in order to have a multiple of window_size
            ary=sample[:, : (time_len // self.window_size) * self.window_size],
            # Number of windows
            indices_or_sections=(time_len // self.window_size),
            axis=-1,
        )

        # Stack the list into a tensor of shape (40, C, window_size)
        X_ts = torch.tensor(np.stack(X_ts, 0), device=self.device)
        # Encode the windows into a Z-vector.
        encodings = self.forward(X_ts)

        # Select a random time step t, spliting the sample into past and future.
        # t is in the range [2, len(encodings) - 2], thus ensuring that "past"
        # and "future" have at least 2 elements.
        random_t = np.random.randint(2, len(encodings) - 2)

        # Split the encodings into "past" and "future"
        # Pick 10 elements before the random_t and 1 element after it
        # Past shape = (S, encoding_size), where 2 < S < 12
        past = encodings[max(0, random_t - 10) : random_t + 1]
        # Add the batch dimension (batch=1).
        # Past shape = (1, S, encoding_size)
        past = past.unsqueeze(0)

        # Generate the context vector (c_t) using the "past" representations.
        _, c_t = self.auto_regressor(past)
        # Flatten it to pass to a linear layer
        c_t = c_t.squeeze(1).squeeze(0)  # Equivalent to c_t.view(-1)
        # Generate the density ratios
        densities = self.density_estimator(c_t)

        # TODO -------- From here, these lines are quite wierd ---------
        log_density_ratios = torch.bmm(
            encodings.unsqueeze(1),
            densities.expand_as(encodings).unsqueeze(-1),
        )

        # Ravel density ratios
        log_density_ratios = log_density_ratios.view(
            -1,
        )

        # r is a set with all time steps except the random_t and its neighbors
        # (random_t-1, random_t+1)
        r = set(range(0, random_t - 2))
        r.update(set(range(random_t + 3, len(encodings))))
        # Select n_size random elements from r
        rnd_n = np.random.choice(list(r), self.n_size)

        # Create a tensor with ``self.n_size`` densitity ratio elements (except
        # the random_t and its neighbors), that constitute the negative samples
        # and the density ratio of the random_t, which is the positive sample.
        X_N = torch.cat(
            [
                log_density_ratios[rnd_n],
                log_density_ratios[random_t + 1].unsqueeze(0),
            ],
            0,
        )
        return X_N

    def training_step(self, batch, batch_idx):
        X_N, loss = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        X_N, loss = self._shared_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        X_N, loss = self._shared_step(batch, batch_idx, "test")
        return loss

    def _shared_step(self, batch, batch_idx, prefix):
        assert len(batch) == 1, "Batch must be 1 sample only"
        assert batch.shape[-1] > 5 * self.window_size, "Sample too short"

        # Generate the encoded representations
        X_N = self._step(batch)
        # Generate the labels
        labels = torch.Tensor([len(X_N) - 1]).to(self.device).long()
        # Calculate the loss
        loss = self.loss_function(X_N.view(1, -1), labels)
        # Log the loss
        self.log(
            f"{prefix}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        return X_N, loss

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
            "n_size": self.n_size,
        }


def build_cpc(
    encoding_size: int = 150,
    in_channel: int = 6,
    gru_hidden_size: int = 100,
    gru_num_layers: int = 1,
    gru_bidirectional: bool = True,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    window_size: int = 4,
    n_size: int = 5,
) -> CPC:
    """Builds a default CPC model. This function aid in the creation of a CPC
    model, by setting the default values of the parameters.

    Parameters
    ----------
    encoding_size : int, optional
        Size of the encoding (output of the linear layer). This is the size of
        the representation.
    in_channel : int, optional
        Number of input features (e.g. 6 for HAR data in MotionSense Dataset)
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
    learning_rate : float, optional
        The learning rate of the optimizer.
    weight_decay : float, optional
        The weight decay of the optimizer.
    window_size : int, optional
        Size of the input windows (X_t) to be fed to the encoder
    n_size : int, optional
        Number of negative samples to be used in the contrastive loss
        (steps to predict)

    Returns
    -------
    CPC
        The CPC model
    """
    encoder = GRUEncoder(
        hidden_size=gru_hidden_size,
        in_channel=in_channel,
        encoding_size=encoding_size,
        num_layers=gru_num_layers,
        dropout=dropout,
        bidirectional=gru_bidirectional,
    )

    density_estimator = torch.nn.Linear(encoding_size, encoding_size)

    auto_regressor = torch.nn.GRU(
        input_size=encoding_size,
        hidden_size=encoding_size,
        batch_first=True,
    )

    model = CPC(
        encoder=encoder,
        density_estimator=density_estimator,
        auto_regressor=auto_regressor,
        lr=learning_rate,
        weight_decay=weight_decay,
        window_size=window_size,
        n_size=n_size,
    )

    return model
