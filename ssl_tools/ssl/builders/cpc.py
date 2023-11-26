# import torch

# from ssl_tools.networks.layers.gru import GRUEncoder
# from ssl_tools.ssl.system.cpc import CPC, CPC_Classifier

# from ssl_tools.ssl.builders.common import StateClassifier


# def build_cpc_pretext(
#     encoding_size: int = 10,
#     window_size: int = 4,
#     learning_rate: float = 1e-3,
#     weight_decay: float = 1e-5,
# ):
#     encoder = GRUEncoder(encoding_size=encoding_size)
#     density_estimator = torch.nn.Linear(encoding_size, encoding_size)
#     auto_regressor = torch.nn.GRU(
#         input_size=encoding_size, hidden_size=encoding_size, batch_first=True
#     )
#     cpc_model = CPC(
#         encoder=encoder,
#         density_estimator=density_estimator,
#         auto_regressor=auto_regressor,
#         window_size=window_size,
#         lr=learning_rate,
#         weight_decay=weight_decay,
#     )
#     return cpc_model


# def build_cpc_classifier(encoder, encoding_size: int = 10, n_classes: int = 6):
#     classifier = StateClassifier(input_size=encoding_size, n_classes=n_classes)
#     cpc_classifier = CPC_Classifier(encoder, classifier)
#     return cpc_classifier


# def build_cpc(encoding_size: int = 10, n_classes: int = 6):
#     encoder = build_cpc_pretext(encoding_size)
#     classifier = build_cpc_classifier(encoder, encoding_size, n_classes)
#     return encoder, classifier
