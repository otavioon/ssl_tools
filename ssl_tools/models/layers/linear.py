import torch


class Discriminator(torch.nn.Module):
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
        super(Discriminator, self).__init__()
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
        return self.model(x).view((-1,))


# class StateClassifier(torch.nn.Module):
#     def __init__(self, input_size: int = 10, n_classes: int = 6):
#         """Simple discriminator network.

#         It is composed by:
#             - BatchNorm1d(``input_size``)
#             - Linear(``input_size``, ``n_classes``)

#         Parameters
#         ----------
#         input_size : int, optional
#             Size of the input sample, by default 10
#         n_classes : int, optional
#             Number of output classes (output_size), by default 1
#         """
#         super(StateClassifier, self).__init__()
#         self.input_size = input_size
#         self.n_classes = n_classes

#         # Defines the model
#         self.model = torch.nn.Sequential(
#             torch.nn.BatchNorm1d(self.input_size),
#             torch.nn.Linear(self.input_size, self.n_classes),
#         )
#         # Init the weights of linear layers with xavier uniform method
#         torch.nn.init.xavier_uniform_(self.model[1].weight)

#     def forward(self, x):
#         return self.model(x)


class StateClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size1=64,
        hidden_size2=64,
        n_classes=6,
        dropout_prob=0,
    ):
        super(StateClassifier, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size1), torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size1, hidden_size2), torch.nn.ReLU()
        )
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size2, n_classes), torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        return out


class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_size: int = 2 * 128, num_classes: int = 2):
        """Simple discriminator network.

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
