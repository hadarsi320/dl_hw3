import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from modules.joint_vae import JointVAE


def interpolate_gif(vae: JointVAE, filename, x_1, x_2, n=100):
    z_1 = vae.hidden_to_latent(x_1.unsqueeze(0))
    z_2 = vae.hidden_to_latent(x_2.unsqueeze(0))

    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])

    interpolate_list = vae.decode(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy() * 255

    images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1]  # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)


def plot_metrics(path, metrics: dict):
    # reconstruction loss
    reconst_losses = metrics["reconstruction_loss"]
    plt.plot(reconst_losses, color="black")
    plt.title("Reconstruction Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction Loss")
    plt.savefig(f"{path}/reconstruction_loss.jpg", dpi=140)
    plt.clf()

    # KL divergences
    cont_kls = metrics["continuous_kl"]
    disc_kls = metrics["disc_kl"]
    plt.plot(cont_kls, color="red", label="continuous")
    plt.plot(disc_kls, color="blue", label="discrete")
    plt.title("KL Divergence vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("KL Divergence")
    plt.legend()
    plt.savefig(f"{path}/kl.jpg", dpi=140)
    plt.clf()