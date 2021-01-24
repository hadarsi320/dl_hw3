import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from modules.joint_vae import JointVAE


"""
Hyperparameters (from paper)
    C.4  CelebA
    Latent distribution: 32 continuous, 1 10-dimensional discrete
    Optimizer: Adam with learning rate 5e-4
    Batch size: 64
    Epochs: 100
    Cz: Increased linearly from 0 to 50 in 100000 iterations
    Cc: Increased linearly from 0 to 10 in 100000 iterations
"""


def train_vae(model: JointVAE, dataloader, num_epochs, optimizer, gamma, C_cont, C_disc, device):
    iteration = 0
    for epoch in range(num_epochs):
        loss_list, iteration = train_epoch(model, dataloader, optimizer, gamma, C_cont, C_disc, iteration, device)
        print(f'Epoch {epoch + 1}: Average loss {np.mean(loss_list)}')


def train_epoch(model, dataloader, optimizer, gamma, C_cont, C_disc, iteration, device):
    loss_list = []
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        iter_loss = train_iteration(model, optimizer, gamma, C_cont, C_disc, iteration, batch)
        loss_list.append(iter_loss.item())
        iteration += 1
    return loss_list, iteration


def train_iteration(model, optimizer: optim.Optimizer, gamma, C_cont, C_disc, iteration, batch):
    optimizer.zero_grad()
    reconst = model(batch)

    reconst_img = reconst[1].detach().cpu().permute(1, 2, 0).numpy()
    matplotlib.pyplot.imshow(reconst_img)
    matplotlib.pyplot.show()

    loss = compute_loss(model, batch, reconst, gamma, C_cont(iteration), C_disc(iteration))
    loss.backward()
    optimizer.step()
    return loss


def compute_loss(model: JointVAE, batch, reconst, gamma, C_cont, C_disc):
    # TODO debug loss values
    # there might be errors which cause imbalance, maybe add some averages
    C_disc = min(C_disc, model.get_max_disc_capacity())
    loss = F.binary_cross_entropy(reconst, batch, reduction='sum') / reconst.size(0)
    if model.is_continuous:
        loss += gamma * torch.abs(model.continuous_kl - C_cont)
    if model.is_discrete:
        loss += gamma * torch.abs(model.discrete_kl - C_disc)
    return loss
