import torch


class Discriminator(torch.nn.Module):
    def __init__(self, input_size: int = 10):
        super(Discriminator, self).__init__()
        self.input_size = input_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(2 * self.input_size, 4 * self.input_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4 * self.input_size, 1),
        )

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))


class StateClassifier(torch.nn.Module):
    def __init__(self, input_size: int = 10, n_classes: int = 6):
        super(StateClassifier, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.normalize = torch.nn.BatchNorm1d(self.input_size)
        self.nn = torch.nn.Linear(self.input_size, self.n_classes)
        torch.nn.init.xavier_uniform_(self.nn.weight)

    def forward(self, x):
        x = self.normalize(x)
        logits = self.nn(x)
        return logits


class SimpleClassifier(torch.nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(2 * 128, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        emb_flatten = x.reshape(x.shape[0], -1)
        x = self.fc(emb_flatten)
        x = torch.sigmoid(x)
        y = self.fc2(x)
        return y
