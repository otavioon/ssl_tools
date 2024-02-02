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


class TNCPredictionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim1: int = 64,
        hidden_dim2: int = 64,
        output_dim: int = 6,
        dropout_prob: float = 0,
    ):
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim1,
                    None,
                    torch.nn.ReLU(),
                ),
                (
                    hidden_dim1,
                    hidden_dim2,
                    None,
                    torch.nn.Sequential(
                        torch.nn.ReLU(), torch.nn.Dropout(p=dropout_prob)
                    ),
                ),
                (
                    hidden_dim2,
                    output_dim,
                    None,
                    torch.nn.Softmax(dim=1),
                ),
            ]
        )


class CPCPredictionHead(TNCPredictionHead):
    pass
