import torch

from ssl_tools.ssl.system.tfc import TFC, TFC_classifier
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from ssl_tools.losses.nxtent import NTXentLoss_poly

from ssl_tools.ssl.builders.common import SimpleClassifier

def build_tfc_pretext(
    length_alignment: int = 178,
    use_cosine_similarity: bool = True,
    temperature: float = 0.5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    time_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            length_alignment, dim_feedforward=2 * length_alignment, nhead=2
        ),
        num_layers=2,
    )
    frequency_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            length_alignment, dim_feedforward=2 * length_alignment, nhead=2
        ),
        num_layers=2,
    )

    time_projector = torch.nn.Sequential(
        torch.nn.Linear(length_alignment, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )
    frequency_projector = torch.nn.Sequential(
        torch.nn.Linear(length_alignment, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )

    nxtent = NTXentLoss_poly(
        batch_size=batch_size,
        temperature=temperature,
        use_cosine_similarity=use_cosine_similarity,
    )

    tfc_model = TFC(
        time_encoder=time_encoder,
        frequency_encoder=frequency_encoder,
        time_projector=time_projector,
        frequency_projector=frequency_projector,
        nxtent_criterion=nxtent,
        lr=learning_rate,
    )


def build_tfc_classifier(
    encoder,
    n_classes: int = 6,
    use_cosine_similarity: bool = True,
    temperature: float = 0.5,
    batch_size: int = 32,
):
    classifier = SimpleClassifier(num_classes=n_classes)
    nxtent = NTXentLoss_poly(
        batch_size=batch_size,
        temperature=temperature,
        use_cosine_similarity=use_cosine_similarity,
    )
    tfc_classifier = TFC_classifier(
        tfc_model=encoder, classifier=classifier, nxtent_criterion=None
    )

    return tfc_classifier


def build_tfc(
    length_alignment: int = 178,
    use_cosine_similarity: bool = True,
    temperature: float = 0.5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    n_classes: int = 6,
):
    encoder = build_tfc_pretext(
        length_alignment=length_alignment,
        use_cosine_similarity=use_cosine_similarity,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    classifier = build_tfc_classifier(
        encoder,
        n_classes=n_classes,
        use_cosine_similarity=use_cosine_similarity,
        temperature=temperature,
        batch_size=batch_size,
    )
    return encoder, classifier
