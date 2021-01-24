import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

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


def train_vae(model: JointVAE, model_name, log_dir, dataloader, num_epochs, optimizer, gamma, C_cont, C_disc, device):
    iteration = 0
    metrics = []  # will hold a dict for each epoch with the metrics
    image_grids = []  # will hold random 16 images to see progress
    for epoch in range(num_epochs):
        epoch_metrics = {
            "total_loss": [],
            "reconstruction_loss": [],
            "continuous_kl": [],
            "disc_kl": [],
        }
        epoch_images = []
        for batch_idx, batch in tqdm(enumerate(dataloader), desc="batch"):
            batch = batch.to(device)

            optimizer.zero_grad()
            reconst = model(batch)

            if batch_idx % 1e4 == 0:
                orig_img = batch[0].detach().cpu().permute(1, 2, 0).numpy()  # redundant after 1 epoch
                reconst_img = reconst[0].detach().cpu().permute(1, 2, 0).numpy()
                epoch_images.append((orig_img, reconst_img))

            iter_loss, reconstruction_loss, continuous_kl, disc_kl = compute_loss(
                model, batch, reconst, gamma, C_cont(iteration), C_disc(iteration)
            )
            iter_loss.backward()
            optimizer.step()

            epoch_metrics["iter_loss"].append(iter_loss.item())
            epoch_metrics["reconstruction_loss"].append(reconstruction_loss.item())
            epoch_metrics["continuous_kl"].append(continuous_kl.item())
            epoch_metrics["disc_kl"].append(disc_kl.item())

            iteration += 1

        report_metrics(epoch, epoch_metrics)
        metrics.append(epoch_metrics)
        image_grids.append(make_image_grid(epoch_images))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./{log_dir}/checkpoint_{epoch}.pt")

    return metrics, image_grids


def report_metrics(epoch, metrics):
    print(f"Epoch: {epoch}")
    for metric_name, value in metrics.items():
        print(f"\t{metric_name:20} {value:.3f}")


def make_image_grid(images, nrow=4):
    return torchvision.utils.make_grid(torch.cat(images[:nrow ** 2]), nrow=nrow).permute(1, 2, 0).detach().numpy()


def compute_loss(model: JointVAE, batch, reconst, gamma, C_cont, C_disc):
    # there might be errors which cause imbalance, maybe add some averages
    C_disc = min(C_disc, model.get_max_disc_capacity())
    reconstruction_loss = F.binary_cross_entropy(reconst, batch, reduction='sum') / reconst.size(0)

    loss = reconstruction_loss
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


def tracked_images():
    w = 28
    img = []
    for i, z2 in enumerate(np.linspace(r1[1], r1[0], n)):
        for j, z1 in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[z1, z2]]).to(device)
            x_hat = autoencoder.decoder(z)
            img.append(x_hat)

    img = torch.cat(img)
    img = torchvision.utils.make_grid(img, nrow=12).permute(1, 2, 0).detach().numpy()
    plt.imshow(img, extent=[*r0, *r1])
