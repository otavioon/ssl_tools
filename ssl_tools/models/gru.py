import torch

class GRUEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 100,
        in_channel: int = 6,
        encoding_size: int = 10,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device
        
        self.rnn = torch.nn.GRU(
            input_size=self.in_channel,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
            bidirectional=bidirectional,
        ).to(device)

        self.nn = torch.nn.Linear(
            self.hidden_size * (int(self.bidirectional) + 1), self.encoding_size
        ).to(device)

    def forward(self, x):
        x = x.permute(2, 0, 1)

        past = torch.zeros(
            self.num_layers * (int(self.bidirectional) + 1),
            x.shape[1],
            self.hidden_size,
            device=self.device,
        )

        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.nn(out[-1].squeeze(0))
        return encodings
