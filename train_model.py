import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from modules.joint_vae import JointVAE


def train_vae(model: JointVAE, dataloader, num_epochs, optimizer, gamma, C_cont, C_disc, device):
    iteration = 0
    for epoch in range(num_epochs):
        loss_list, iteration = train_epoch(model, dataloader, optimizer, gamma, C_cont, C_disc, iteration, device)
        print(f'Epoch {epoch + 1}: Average loss {np.mean(loss_list)}')


def train_epoch(model, dataloader, optimizer, gamma, C_cont, C_disc, iteration, device):
    loss_list = []
    for batch, _ in dataloader:
        batch = batch.to(device)
        iter_loss = train_iteration(model, optimizer, gamma, C_cont, C_disc, iteration, batch)
        loss_list.append(iter_loss.item())
        iteration += 1
    return loss_list, iteration


def train_iteration(model, optimizer: optim.Optimizer, gamma, C_cont, C_disc, iteration, batch):
    optimizer.zero_grad()
    reconst = model(batch)
    loss = compute_loss(model, batch, reconst, gamma, C_cont(iteration), C_disc(iteration))
    loss.backward()
    optimizer.step()
    return loss


def compute_loss(model: JointVAE, batch, reconst, gamma, C_cont, C_disc):
    # TODO debug loss values
    # there might be errors which cause imbalance, maybe add some averages
    C_disc = min(C_disc, model.get_max_disc_capacity())
    loss = F.binary_cross_entropy(reconst, batch, reduction='sum')
    if model.is_continuous:
        loss += gamma * torch.abs(model.continuous_kl - C_cont)
    if model.is_discrete:
        loss += gamma * torch.abs(model.discrete_kl - C_disc)
    return loss
