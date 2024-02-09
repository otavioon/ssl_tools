from typing import Tuple
import torch
from torch import nn
import torch.nn as nn


from ssl_tools.models.nets.simple import SimpleReconstructionNet
from ssl_tools.losses.contrastive_loss import ContrastiveLoss


# Equivalent Keras Code
# class ConvolutionalAutoEncoder:
#     def __init__(
#         self, sequence_length: int, num_features: int, learning_rate: float = 0.0001
#     ):
#         self.sequence_length = sequence_length
#         self.num_features = num_features
#         self.learning_rate = learning_rate
#         self.model = self.build_model()

#     def build_model(self):
#         model = keras.Sequential(
#             [
#                 layers.Input(shape=(self.sequence_length, self.num_features)),
#                 layers.Conv1D(
#                     filters=32,
#                     kernel_size=7,
#                     padding="same",
#                     strides=2,
#                     activation="relu",
#                 ),
#                 layers.Dropout(rate=0.2),
#                 layers.Conv1D(
#                     filters=16,
#                     kernel_size=7,
#                     padding="same",
#                     strides=2,
#                     activation="relu",
#                 ),
#                 layers.Conv1DTranspose(
#                     filters=16,
#                     kernel_size=7,
#                     padding="same",
#                     strides=2,
#                     activation="relu",
#                 ),
#                 layers.Dropout(rate=0.2),
#                 layers.Conv1DTranspose(
#                     filters=32,
#                     kernel_size=7,
#                     padding="same",
#                     strides=2,
#                     activation="relu",
#                 ),
#                 layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
#             ]
#         )
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
#         return model

class _ConvolutionalAutoEncoder(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int, int] = (1, 16)):
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_shape[0], 32, kernel_size=7, stride=2, padding=3
        )
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=7, stride=2, padding=3)
        self.conv_transpose1 = nn.ConvTranspose1d(
            16, 16, kernel_size=7, stride=2, padding=3, output_padding=1
        )
        self.dropout2 = nn.Dropout(0.2)
        self.conv_transpose2 = nn.ConvTranspose1d(
            16, 32, kernel_size=7, stride=2, padding=3, output_padding=1
        )
        self.conv_transpose3 = nn.ConvTranspose1d(
            32, input_shape[0], kernel_size=7, padding=3
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv_transpose1(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        return x
    

class _ConvolutionalAutoEncoder2D(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 4, 4)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=1)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, input_shape[0], kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.upsample1(x)
        x = torch.relu(self.conv4(x))
        x = self.upsample2(x)
        x = torch.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x


class ConvolutionalAutoEncoder(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (1, 16),
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            backbone=_ConvolutionalAutoEncoder(input_shape=input_shape),
            learning_rate=learning_rate,
            loss_fn=nn.MSELoss(),
        )
        self.input_shape = input_shape


class ConvolutionalAutoEncoder2D(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 4, 4),
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            backbone=_ConvolutionalAutoEncoder2D(input_shape=input_shape),
            learning_rate=learning_rate,
            loss_fn=nn.MSELoss(),
        )
        self.input_shape = input_shape


class ContrastiveConvolutionalAutoEncoder(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (1, 16),
        learning_rate: float = 1e-3,
        margin: float = 1.0,
    ):
        super().__init__(
            backbone=_ConvolutionalAutoEncoder(input_shape=input_shape),
            learning_rate=learning_rate,
            loss_fn=ContrastiveLoss(margin),
        )

# Equivalent Keras code
# class ContrastiveConvolutionalAutoEncoder:
#     def __init__(
#         self,
#         input_shape: tuple,
#         learning_rate: float = 0.0001,
#         margin: float = 1.0,
#     ):
#         self.input_shape = input_shape
#         self.learning_rate = learning_rate
#         self.margin = margin
#         self.model = self.build_model()

#     def contrastive_loss(self, y_true, y_pred):
#         """Calculates the constrastive loss.

#         Arguments:
#             y_true: List of labels, each label is of type float32.
#             y_pred: List of predictions of same length as of y_true,
#                     each label is of type float32.

#         Returns:
#             A tensor containing constrastive loss as floating point value.
#         """

#         square_pred = tf.math.square(y_pred)
#         margin_square = tf.math.square(tf.math.maximum(self.margin - (y_pred), 0))
#         return tf.math.reduce_mean(
#             (1 - y_true) * square_pred + (y_true) * margin_square
#         )

#     def build_model(self):
#         optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

#         model = keras.Sequential()
#         model.add(
#             layers.Conv2D(
#                 32,
#                 (3, 3),
#                 activation="relu",
#                 padding="same",
#                 input_shape=self.input_shape,
#             )
#         )
#         model.add(layers.MaxPool2D(padding="same"))
#         model.add(layers.Conv2D(16, (3, 3), activation="relu", padding="same"))
#         model.add(layers.MaxPool2D(padding="same"))
#         model.add(layers.Conv2D(8, (3, 3), activation="relu", padding="same"))
#         model.add(layers.UpSampling2D())
#         model.add(layers.Conv2D(16, (3, 3), activation="relu", padding="same"))
#         model.add(layers.UpSampling2D())
#         model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
#         model.add(
#             layers.Conv2D(
#                 self.input_shape[-1], (3, 3), activation="sigmoid", padding="same"
#             )
#         )
#         model.compile(loss=self.contrastive_loss, optimizer=optimizer)

#         return model

class ContrastiveConvolutionalAutoEncoder2D(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (4, 4, 1),
        learning_rate: float = 1e-3,
        margin: float = 1.0,
    ):
        super().__init__(
            backbone=_ConvolutionalAutoEncoder2D(input_shape=input_shape),
            learning_rate=learning_rate,
            loss_fn=ContrastiveLoss(margin),
        )
        self.input_shape = input_shape