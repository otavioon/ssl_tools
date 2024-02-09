import torch
import lightning as L
import torch.nn as nn

from typing import Tuple

from ssl_tools.models.nets.simple import SimpleReconstructionNet  


# Equivalent Keras Code
# class LSTMAutoEncoder:
#     def __init__(self, time_units: int, lstm_units: int, 
#             learning_rate: float = 0.0001):
#         self.time_units = time_units
#         self.lstm_units = lstm_units
#         self.learning_rate = learning_rate
#         self.model = self.build_model()

#     def build_model(self):
#         model = keras.Sequential()
#         # shape [batch, time, features] => [batch, time, lstm_units]
#         model.add(keras.layers.LSTM(
#             units=128,
#             input_shape=(self.time_units, self.lstm_units), # univariate input
#             return_sequences=True)
#         )
#         #model.add(keras.layers.Dropout(rate=0.2))
#         model.add(keras.layers.LSTM(units=64, return_sequences=False))
#         model.add(keras.layers.RepeatVector(n=self.time_units))
#         model.add(keras.layers.LSTM(units=64, return_sequences=True))
#         model.add(keras.layers.LSTM(units=128, return_sequences=True))
#         #model.add(keras.layers.Dropout(rate=0.2))
#         # shape => [batch, time, features]
#         model.add(keras.layers.TimeDistributed(
#             keras.layers.Dense(units=self.lstm_units))
#         ) # univariate output
#         model.compile(
#             loss=tf.losses.MeanSquaredError(),
#             optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
#             metrics=[tf.metrics.MeanSquaredError()]
#         )

#         return model


class _LSTMAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (16, 1),
    ):
        super().__init__()
        self.input_shape = input_shape
        self.lstm1 = nn.LSTM(
            input_size=input_shape[1], hidden_size=128, batch_first=True
        )
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.repeat_vector = nn.Linear(64, 64 * input_shape[0])
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.time_distributed = nn.Linear(128, input_shape[1])

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = x.unsqueeze(1).repeat(1, self.input_shape[0], 1)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.time_distributed(x)
        return x
    
    
class LSTMAutoencoder(SimpleReconstructionNet):
    def __init__(
        self,
        input_shape: Tuple[int, int] = (16, 1),
        learning_rate: float = 1e-3,
    ):
        """Create a LSTM Autoencoder model

        Parameters
        ----------
        input_shape : Tuple[int, int], optional
            The shape of the input. The first element is the sequence length and the second is the number of features, by default (16, 1)
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 1e-3
        """
        super().__init__(
            backbone=_LSTMAutoEncoder(input_shape),
            learning_rate=learning_rate,
            loss_fn=nn.MSELoss(),
        )

