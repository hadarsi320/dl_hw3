import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from modules.joint_vae import JointVAE


def compute_loss(model: JointVAE, batch, reconst, gamma, C_cont, C_disc):
    C_disc = min(C_disc, model.get_max_disc_capacity())
    reconstruction_loss = F.binary_cross_entropy(reconst, batch, reduction='sum') / reconst.size(0)

    loss = reconstruction_loss.clone()
    if model.is_continuous:
        continuous_kl = model.continuous_kl
        loss += gamma * torch.abs(continuous_kl - C_cont)
    else:
        continuous_kl = np.nan
    if model.is_discrete:
        disc_kl = model.discrete_kl
        loss += gamma * torch.abs(disc_kl - C_disc)
    else:
        disc_kl = np.nan
    return loss, reconstruction_loss, continuous_kl, disc_kl


@torch.no_grad()
def eval_model_reconstruction(model: JointVAE, dataloader):
    loss = 0
    for batch, _ in tqdm(dataloader):
        batch = batch.cuda()
        reconst = model(batch)
        loss += F.binary_cross_entropy(reconst, batch, reduction='sum').item()
    return loss / len(dataloader.dataset)
