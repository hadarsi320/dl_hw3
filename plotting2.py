import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from celeba_dataset import CelebADataset
from modules.joint_vae import JointVAE

CONT_DIM = 32
VIZ_DIR = 'viz'


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


def get_latent_labels(vae: JointVAE, dataloader, device='cuda'):
    latents = []
    labels = []
    for batch, batch_labels in tqdm(dataloader, desc='Encoding Dataset'):
        latents.append(vae.get_latent(batch.to(device)))
        labels.append(batch_labels.to(device))
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    return latents, labels


def get_latent_boundaries(latents, i):
    return latents[:, i].min().item(), latents[:, i].max().item()


def vis2(vae: JointVAE, dataloader, image_id, ncols=10):
    latents, _ = get_latent_labels(vae, dataloader)
    latent_boundaries = [get_latent_boundaries(latents, i) for i in range(CONT_DIM)]
    img_latent = latents[image_id]  # get an arbitrary image's latent space
    image_grids = []
    col_spaces = []

    disc_dim = len(img_latent) - CONT_DIM

    # for each direction in the latent dim
    for i in range(len(img_latent)):
        if i == CONT_DIM:
            break

        z_i_min = latent_boundaries[i][0]
        z_i_max = latent_boundaries[i][1]
        tmp = img_latent.clone()
        tmp[CONT_DIM:] = torch.zeros_like(tmp[CONT_DIM:])

        img_grid = []
        col_space = []
        # create img_grid rows: iterate discrete dim
        for disc_idx in range(CONT_DIM, CONT_DIM + disc_dim):
            tmp[disc_idx] = 1.0

            # create img_grid cols: set all variables besides the current z_i
            col_space = [round(n, 3) for n in np.linspace(z_i_min, z_i_max, num=ncols)]
            for val in col_space:
                tmp[i] = val
                img_grid.append(vae.decode(tmp.unsqueeze(0)))

            tmp[disc_idx] = 0.0

        img_grid = torch.cat(img_grid)
        img_grid = torchvision.utils.make_grid(img_grid, nrow=ncols).permute(1, 2, 0).cpu().detach().numpy()
        image_grids.append(img_grid)
        col_spaces.append(col_space)

    return image_grids, col_spaces


def main():
    train_dataset = CelebADataset('celeba_resized', limit=100)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=6
    )

    checkpoint = torch.load('logs/full-150epochs/checkpoint_149.pt')
    model = JointVAE(latent_spec={'cont': CONT_DIM, 'disc': [10]})
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    model.eval()

    image_id = 0
    ncols = 10
    image_grids, col_spaces = vis2(model, train_loader, image_id, ncols)

    for i, (img, col_space) in enumerate(zip(image_grids, col_spaces)):
        plt.imshow(img)
        plt.title(f"Image ID: {image_id}\nLatent variable: z_{i}")

        plt.xticks(ticks=np.linspace(0, len(img[0]), num=3), labels=[col_space[0], col_space[round(len(col_space) / 2)], col_space[-1]])

        yticks = [n for n in np.linspace(0, len(img), num=10)]
        plt.yticks(ticks=yticks, labels=[f"Discrete {i}" for i in range(10)])
        plt.savefig(f"{VIZ_DIR}/img_{image_id}__z_{i}.jpg", dpi=300)
        plt.clf()


if __name__ == '__main__':
    main()
