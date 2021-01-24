from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self._decoder = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x

    def encode(self, x):
        return self._encoder(x)

    def decode(self, x):
        return self._decoder(x)
