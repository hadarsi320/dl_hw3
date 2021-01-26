from torch import nn
from torch import optim
from tqdm import tqdm


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

    def train_model(self, dataloader, num_epochs,
                    loss_type='MSELoss', learning_rate=1e-3):
        loss_function = getattr(nn, loss_type)()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for _ in tqdm(range(num_epochs), 'Training AE'):
            for [batch] in dataloader:
                optimizer.zero_grad()
                reconst = self(batch)
                loss = loss_function(reconst, batch)
                loss.backward()
                optimizer.step()
