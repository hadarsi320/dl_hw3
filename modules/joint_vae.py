import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

EPS = 1e-20


class JointVAE(nn.Module):
    def __init__(self, latent_spec, temperature=0.67, hard=True, hidden_dim=256):
        super(JointVAE, self).__init__()

        self._hard = hard
        self._temperature = temperature
        self._hidden_dim = hidden_dim
        self._latent_spec = latent_spec
        self._max_disc_capacity = sum([math.log(dim) for dim in self._latent_spec['disc']])

        self.continuous_kl = 0
        self.discrete_kl = 0

        self.is_continuous = 'cont' in self._latent_spec
        self.is_discrete = 'disc' in self._latent_spec

        # Calculate dimensions of latent distribution
        if self.is_continuous:
            self.latent_cont_dim = self._latent_spec['cont']
        else:
            self.latent_cont_dim = 0

        if self.is_discrete:
            self.latent_disc_dim = sum(self._latent_spec['disc'])
            self.num_disc_latents = len(self._latent_spec['disc'])
        else:
            self.latent_disc_dim = 0
            self.num_disc_latents = 0
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 4 * 4, self._hidden_dim),
            nn.ReLU(),
        )

        # Encode parameters of latent distribution
        if self.is_continuous:
            self.fc_mean = nn.Linear(self._hidden_dim, self.latent_cont_dim)
            self.fc_log_var = nn.Linear(self._hidden_dim, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            fc_alphas = []
            for disc_dim in self._latent_spec['disc']:
                fc_alphas.append(nn.Sequential(
                    nn.Linear(self._hidden_dim, disc_dim),
                    nn.Softmax(dim=1)
                ))
            self._fc_alphas = nn.ModuleList(fc_alphas)

        self._decoder = nn.Sequential(
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
            nn.Sigmoid(),  # this scales our predictions to the range of images (0, 1)
        )

    def forward(self, x):
        x = self._encoder(x)
        x = self.hidden_to_latent(x)
        x = self._decoder(x)
        return x

    def get_latent(self, x):
        x = self._encoder(x)
        x = self.hidden_to_latent(x)
        return x

    def decode(self, x):
        return self._decoder(x)

    def hidden_to_latent(self, encoding):
        latent = []
        if self.is_continuous:
            mu = self.fc_mean(encoding)
            log_var = self.fc_log_var(encoding)
            self.continuous_kl = - 0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
            latent.append(self.sample_normal(mu, log_var))

        self.discrete_kl = 0
        if self.is_discrete:
            for fc_alpha in self._fc_alphas:
                alpha = fc_alpha(encoding)
                latent.append(self.sample_gumbel_softmax(alpha))
                # compute discrete KL
                log_dim = math.log(alpha.shape[1])
                mean_neg_entropy = torch.mean(torch.sum(alpha * torch.log(alpha + EPS), dim=1), dim=0)
                self.discrete_kl += log_dim + mean_neg_entropy

        return torch.cat(latent, dim=1)

    def sample_normal(self, mu, log_var):
        if self.training:
            sample = torch.randn_like(log_var)
            return mu + sample * torch.exp(log_var / 2)
        return mu

    def sample_gumbel_softmax(self, alpha):
        if self.training:
            log_alpha = torch.log(alpha + EPS)
            gumbel_noise = utils.sample_gumbel(alpha.shape)
            y_soft = F.softmax((log_alpha + gumbel_noise) / self._temperature, dim=1)

            if self._hard:
                k = torch.argmax(y_soft, dim=-1)
                y_hard = F.one_hot(k, num_classes=alpha.shape[1])
                y = y_hard - y_soft.detach() + y_soft
            else:
                y = y_soft
            return y

        else:
            k = torch.argmax(alpha, dim=-1)
            y = F.one_hot(k, num_classes=alpha.shape[-1])
            return y

    def get_max_disc_capacity(self):
        return self._max_disc_capacity
