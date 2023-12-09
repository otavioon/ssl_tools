from typing import Any, Tuple
import lightning as L
import torch

from ssl_tools.utils.configurable import Configurable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from ssl_tools.losses.nxtent import NTXentLoss_poly


class TFCHead(torch.nn.Module):
    def __init__(self, input_size: int = 2 * 128, num_classes: int = 2):
        """Simple discriminator network, used as the head of the TFC model.

        Parameters
        ----------
        input_size : int, optional
            Size of the input sample, by default 2*128
        n_classes : int, optional
            Number of output classes (output_size), by default 2
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # Defines the model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, self.num_classes),
        )
        # Init the weights of linear layers with xavier uniform method
        # torch.nn.init.xavier_uniform_(self.model[0].weight)
        # torch.nn.init.xavier_uniform_(self.model[2].weight)

    def forward(self, x):
        emb_flatten = x.reshape(x.shape[0], -1)
        return self.model(emb_flatten)


class TFC(L.LightningModule, Configurable):
    def __init__(
        self,
        time_encoder: torch.nn.Module,
        frequency_encoder: torch.nn.Module,
        time_projector: torch.nn.Module,
        frequency_projector: torch.nn.Module,
        nxtent_criterion: torch.nn.Module,
        learning_rate: float = 1e-3,
        loss_lambda: float = 0.2,
        permute_input: tuple = None,
    ):
        """Implements the Time-Frequency Contrastive model, as described in:
        Zhang, Xiang, et al. "Self-supervised contrastive pre-training for time
        series via time-frequency consistency." Advances in Neural Information
        Processing Systems 35 (2022): 3988-4003.

        Parameters
        ----------
        time_encoder : torch.nn.Module
            The encoder for the time-domain data. It is usually a convolutional
            encoder such as a resnet1D or a transformer.
        frequency_encoder : torch.nn.Module
            The encoder for the frequency-domain data. It is usually a
            convolutional encoder such as a resnet1D or a transformer.
        time_projector : torch.nn.Module
            The projector for the time-domain data. Usually the projector is a
            linear layer with the desired output dimensionality.
        frequency_projector : torch.nn.Module
            The projector for the frequency-domain data. Usually the projector
            is a linear layer with the desired output dimensionality.
        nxtent_criterion : torch.nn.Module
            The Normalized Temperature-scaled Cross Entropy Loss.
        learning_rate : float, optional
            The learning rate for the optimizer, by default 1e-3
        loss_lambda : float, optional
            The consistency threshold, by default 0.2
        """
        super().__init__()

        self.time_encoder = time_encoder
        self.time_projector = time_projector
        self.frequency_encoder = frequency_encoder
        self.frequency_projector = frequency_projector
        self.nxtent_criterion = nxtent_criterion
        self.learning_rate = learning_rate
        self.loss_lambda = loss_lambda
        self.permute_input = permute_input

    def forward(
        self, x_in_t: torch.Tensor, x_in_f: torch.Tensor
    ) -> torch.Tensor:
        """Generate the final representation of the model.

        Parameters
        ----------
        x_in_t : torch.Tensor
            The time-domain data.
        x_in_f : torch.Tensor
            The frequency-domain data.

        Returns
        -------
        torch.Tensor
            The final representation of the model (z_t, z_f concatenated)
        """
        if self.permute_input is not None:
            x_in_t = x_in_t.permute(*self.permute_input)
            x_in_f = x_in_f.permute(*self.permute_input)
        h_t, z_t, h_f, z_f = self._generate_representations(x_in_t, x_in_f)
        return torch.cat((z_t, z_f), dim=1)

    def training_step(self, batch, batch_idx):
        # Batch is a 5-element tuple with the following elements:
        # - The original time-domain signal
        # - The label of the signal
        # - Time augmented signal
        # - The frequency signal
        # - The frequency augmented signal
        data, labels, aug1, data_f, aug1_f = batch
        (h_t, z_t, h_f, z_f), loss = self._shared_step(
            data, aug1, data_f, aug1_f, "train"
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # Batch is a 5-element tuple with the following elements:
        # - The original time-domain signal
        # - The label of the signal
        # - Time augmented signal
        # - The frequency signal
        # - The frequency augmented signal
        data, labels, aug1, data_f, aug1_f = batch
        (h_t, z_t, h_f, z_f), loss = self._shared_step(
            data, aug1, data_f, aug1_f, "val"
        )
        return loss

    def test_step(self, batch, batch_idx):
        # Batch is a 5-element tuple with the following elements:
        # - The original time-domain signal
        # - The label of the signal
        # - Time augmented signal
        # - The frequency signal
        # - The frequency augmented signal
        data, labels, aug1, data_f, aug1_f = batch
        (h_t, z_t, h_f, z_f), loss = self._shared_step(
            data, aug1, data_f, aug1_f, "test"
        )
        return loss

    def _generate_representations(
        self, x_in_t: torch.Tensor, x_in_f: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the intermediate representations of the model.

        Parameters
        ----------
        x_in_t : torch.Tensor
            A tensor with the time-domain data. Usually has shape: (B, C, T),
            where B is the batch size, C is the number of channels and T is the
            number of time steps.
        x_in_f : _type_
            A tensor with the frequency-domain data. Usually has shape:
            (B, C, F), where B is the batch size, C is the number of channels
            F is the number of frequency bins.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A 4-tuple with the intermediate representations of the model:
            (h_time, z_time, h_freq, z_freq).
        """
        # Encodes the time-domain data. It is usually a convolutional encoder
        # such as a resnet1D or a transformer
        x = self.time_encoder(x_in_t)
        # Reshape the input to be (B, C*T)
        h_time = x.reshape(x.shape[0], -1)
        # Project the time-domain data. Usually the projector is a linear layer
        # with the desired output dimensionality
        z_time = self.time_projector(h_time)

        # Encodes the frequency-domain data. It is usually a convolutional
        # encoder such as a resnet1D or a transformer
        f = self.frequency_encoder(x_in_f)
        # Reshape the input to be (B, C*F)
        h_freq = f.reshape(f.shape[0], -1)
        # Project the frequency-domain data. Usually the projector is a linear
        # layer with the desired output dimensionality
        z_freq = self.frequency_projector(h_freq)

        # Concatenate the time and frequency representations (final
        # representation)
        return h_time, z_time, h_freq, z_freq

    def _shared_step(
        self,
        data: torch.Tensor,
        aug1: torch.Tensor,
        data_f: torch.Tensor,
        aug1_f: torch.Tensor,
        stage: str,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Compute the representations and the loss.

        Parameters
        ----------
        data : torch.Tensor
            The original time-domain data
        aug1 : torch.Tensor
            The augmented time-domain data
        data_f : torch.Tensor
            The original frequency-domain data
        aug1_f : torch.Tensor
            The augmented frequency-domain data
        stage : str
            Stage of the training (train, val, test)

        Returns
        -------
        Tuple[ Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, ]
            Returns a 2-element tuple. The first element is a 4-element tuple
            with the intermediate representations of the model: (h_time,
            z_time, h_freq, z_freq). The second element is the loss.
        """
        if self.permute_input is not None:
            data = data.permute(*self.permute_input)
            aug1 = aug1.permute(*self.permute_input)
            data_f = data_f.permute(*self.permute_input)
            aug1_f = aug1_f.permute(*self.permute_input)
        
        # Get intermetiate representations for non-augmented and augmented data
        # h_* is the intermediate representation of the encoder
        # z_* is the intermediate representation of the projector
        h_t, z_t, h_f, z_f = self._generate_representations(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self._generate_representations(
            aug1, aug1_f
        )

        # Calculate the Normalized Temperature-scaled Cross Entropy Loss for
        # between: encoded representations of non-augmented and augmented tima data
        # and frequency data. Also, between: projected representations of
        # non-augmented and augmented data.
        loss_time = self.nxtent_criterion(h_t, h_t_aug)
        loss_freq = self.nxtent_criterion(h_f, h_f_aug)
        loss_consistency = self.nxtent_criterion(z_t, z_f)
        # Calculate the total loss
        loss = (self.loss_lambda * (loss_time + loss_freq)) + loss_consistency

        # log loss, only to appear on epoch
        # self.log(
        #     f"{stage}_time_loss",
        #     loss_time,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        # self.log(
        #     f"{stage}_freq_loss",
        #     loss_freq,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        # self.log(
        #     f"{stage}_consistency_loss",
        #     loss_consistency,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return (h_t, z_t, h_f, z_f), loss

    def configure_optimizers(self) -> Any:
        learnable_parameters = self.parameters()
        optimizer = torch.optim.Adam(
            learnable_parameters, lr=self.learning_rate
        )
        return optimizer

    def get_config(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "loss_lambda": self.loss_lambda,
        }


def build_tfc_transformer(
    encoding_size: int = 128,
    in_channels: int = 1,
    length_alignment: int = 360,
    use_cosine_similarity: bool = True,
    learning_rate: float = 1e-3,
    temperature: float = 0.5,
) -> TFC:
    """Creates a TFC model with a transformer encoder. This function aids in
    the creation of the TFC model, by providing a transformer encoder and
    projector.

    Parameters
    ----------
        Size of the encoding (output of the linear layer). This is the size of
        the representation.
    in_channels : int, optional
        Number of channels of the input data (e.g. 6 for HAR data in 
        MotionSense Dataset),
    length_alignment : int, optional
        Truncate the features to this value
    use_cosine_similarity : bool, optional
        If True, the cosine similarity will be used instead of the dot product,
        in the NTXentLoss.
    learning_rate : float, optional
        The learning rate for the optimizer.
    temperature : float, optional
        The temperature for the NTXentLoss.

    Returns
    -------
    TFC
        A TFC model with a transformer encoder.
    """
    
    # Instantiate Encoders
    time_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            in_channels,
            dim_feedforward=2 * length_alignment,
            nhead=2,
            batch_first=True,
        ),
        num_layers=2,
    )
    frequency_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            in_channels,
            dim_feedforward=2 * length_alignment,
            nhead=2,
            batch_first=True,
        ),
        num_layers=2,
    )

    # Instantiate Projectors
    time_projector = torch.nn.Sequential(
        torch.nn.Linear(in_channels * length_alignment, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, encoding_size),
    )
    frequency_projector = torch.nn.Sequential(
        torch.nn.Linear(in_channels * length_alignment, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, encoding_size),
    )

    # Instantiate NTXentLoss
    nxtent = NTXentLoss_poly(
        temperature=temperature,
        use_cosine_similarity=use_cosine_similarity,
    )

    # Create the model
    model = TFC(
        time_encoder=time_encoder,
        frequency_encoder=frequency_encoder,
        time_projector=time_projector,
        frequency_projector=frequency_projector,
        nxtent_criterion=nxtent,
        learning_rate=learning_rate,
        permute_input=(0, 2, 1)
    )

    return model
