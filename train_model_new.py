import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def train_vae(model: JointVAE, log_dir, dataloader, num_epochs, optimizer, gamma, C_cont, C_disc, device):
    iteration = 0
    metrics = {  # will hold a dict for each epoch with the metrics
        "loss": [],
        "reconstruction_loss": [],
        "continuous_kl": [],
        "disc_kl": [],
    }
    image_grids = []  # will hold random 16 images to see progress
    for epoch in range(num_epochs):
        epoch_metrics = {
            "iter_loss": [],
            "reconstruction_loss": [],
            "continuous_kl": [],
            "disc_kl": [],
        }
        epoch_images = []
        for batch_idx, (batch, _) in enumerate(dataloader):
            batch = batch.to(device)

            optimizer.zero_grad()
            reconst = model(batch)

            if batch_idx % 1e2 == 0:
                reconst_img = reconst[0].unsqueeze(0)
                epoch_images.append(reconst_img)

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
        metrics["loss"].append(np.mean(epoch_metrics["iter_loss"]))
        metrics["reconstruction_loss"].append(np.mean(epoch_metrics["reconstruction_loss"]))
        metrics["continuous_kl"].append(np.mean(epoch_metrics["continuous_kl"]))
        metrics["disc_kl"].append(np.mean(epoch_metrics["disc_kl"]))

        image_grids.append(make_image_grid(log_dir, epoch, epoch_images))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./{log_dir}/checkpoint_{epoch}.pt")

    return metrics, image_grids


def report_metrics(epoch, metrics):
    string = f"Epoch: {epoch:3d}\t"
    for metric_name, values in metrics.items():
        string += f"\t{metric_name:20} {sum(values) / len(values): 5.3f}"
    print(string)


def make_image_grid(path, epoch, images):
    nrow = np.math.floor(np.sqrt(len(images)))
    img = torchvision.utils.make_grid(torch.cat(images[:nrow ** 2]), nrow=nrow).detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(f"epoch {epoch}")
    plt.savefig(f"{path}/faces_{epoch}.jpg")
    return img


def compute_loss(model: JointVAE, batch, reconst, gamma, C_cont, C_disc):
    # there might be errors which cause imbalance, maybe add some averages
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
