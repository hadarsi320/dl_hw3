import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from modules.joint_vae import JointVAE
from nn_utils import compute_loss


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
        string += f"\t{metric_name:20} {np.mean(values): 5.3f}"
    print(string)


def make_image_grid(path, epoch, images):
    nrow = np.math.floor(np.sqrt(len(images)))
    img = torchvision.utils.make_grid(torch.cat(images[:nrow ** 2]), nrow=nrow).detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(f"epoch {epoch}")
    plt.savefig(f"{path}/faces_{epoch}.jpg")
    plt.clf()
    return img
