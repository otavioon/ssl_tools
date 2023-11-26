# from ssl_tools.ssl_tools.networks.layers.gru import GRUEncoder
# from ssl_tools.ssl.system.tnc import TNC, TNC_Classifier
# from ssl_tools.ssl.builders.common import Discriminator, StateClassifier


# def build_tnc_pretext(
#     encoding_size: int = 10, window_size: int = 4, mc_sample_size: int = 20
# ):
#     discriminator = Discriminator(input_size=encoding_size, device="cuda")
#     encoder = GRUEncoder(encoding_size=encoding_size, device="cuda")
#     tnc_model = TNC(
#         encoder=encoder,
#         discriminator=discriminator,
#         window_size=window_size,
#         mc_sample_size=mc_sample_size,
#     )
#     return tnc_model


# def build_tnc_classifier(encoder, encoding_size: int = 10, n_classes: int = 6):
#     classifier = StateClassifier(input_size=encoding_size, n_classes=n_classes)
#     tnc_classifier = TNC_Classifier(encoder, classifier)
#     return tnc_classifier


# def build_tnc(
#     encoding_size: int = 10,
#     n_classes: int = 6,
#     window_size: int = 4,
#     mc_sample_size: int = 20,
# ):
#     encoder = build_tnc_pretext(encoding_size, window_size, mc_sample_size)
#     classifier = build_tnc_classifier(encoder, encoding_size, n_classes)
#     return encoder, classifier
