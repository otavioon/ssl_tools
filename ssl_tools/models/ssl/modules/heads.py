import torch
from lightly.models.modules.heads import ProjectionHead


class TFCProjectionHead(ProjectionHead):
    def __init__(
        self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128
    ):
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim,
                    torch.nn.BatchNorm1d(256),
                    torch.nn.ReLU(),
                ),
                (hidden_dim, output_dim, None, None),
            ]
        )


class TFCPredictionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 2 * 128,
        hidden_dim: int = 64,
        output_dim: int = 2,
    ):
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim,
                    None,
                    torch.nn.Sigmoid(),
                ),
                (hidden_dim, output_dim, None, None),
            ]
        )
