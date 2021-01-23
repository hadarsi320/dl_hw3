import torch
import torch.nn as nn


class JointVAE(nn.Module):
    def __init__(self, latent_spec):

        super(JointVAE, self).__init__()

        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec

        # Calculate dimensions of latent distribution
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        else:
            self.latent_cont_dim = 0

        if self.is_discrete:
            self.latent_disc_dim = sum(self.latent_spec['disc'])
            self.num_disc_latents = len(self.latent_spec['disc'])
        else:
            self.latent_disc_dim = 0
            self.num_disc_latents = 0
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        self.continuous_kl = 0
        self.discrete_kl = 0

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 4 * 4, self.hidden_dim),
            nn.ReLU(),
        )

        # Encode parameters of latent distribution
        if self.is_continuous:
            self.fc_mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
            self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(self.hidden_dim, disc_dim))
            self.fc_alphas = nn.ModuleList(fc_alphas)

        # Map latent samples to features to be used by generative model
        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 64 * 4 * 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = 5
        # input:
        # (batch, 3, 64, 64)

        # conv2d
        # (batch, 32, 32, 32)

        # conv2d
        # (batch, 32, 16, 16)

        # conv2d
        # (batch, 64, 8, 8)

        # conv2d
        # (batch, 64, 4, 4)

        # flatten start from 1

        # linear
        # (batch, 256)

        # ???
        # sampling
        # reparameterization
        # ???

        # linear
        # (batch, 256)

        # linear
        # (batch, 64 * 4 * 4)

        # flatten transpose (64, 4, 4)

        # convtrans2d
        # (batch, 64, 8, 8)

        # convtrans2d
        # (batch, 32, 16, 16)

        # convtrans2d
        # (batch, 32, 32, 32)

        # convtrans2d
        # (batch, 3, 64, 64)

        x = self.encoder(x)
        # x = self.hidden_to_latent(x)
        x = self.decoder(x)
        return x

    def hidden_to_latent(self, x):
        latent = []
        if self.is_continuous:
            mu = self.fc_mean(x)
            log_var = self.fc_log_var(x)
            self.cont_kl = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            sample = torch.randn_like(log_var)
            latent.append(mu + sample * torch.exp(log_var / 2))
        else:
            self.cont_kl = 0

        if self.is_discrete:
            pass

        return torch.cat(latent)
