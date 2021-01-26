import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

import utils
from celeba_dataset import CelebADataset
from modules.autoencoder import Autoencoder
from modules.joint_vae import JointVAE

from torchvision.transforms import Resize, ToTensor
from PIL import Image


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


# viz #1
def find_correlated_dimensions(cont_latents, labels, label_names):
    scores = {}
    for i in tqdm(range(cont_latents.shape[1] - 1), 'i'):
        for j in range(i + 1, cont_latents.shape[1]):
            for k in range(labels.shape[1]):
                X = cont_latents[:, [i, j]]
                Y = labels[:, k]
                X, Y = utils.under_sample(X, Y, n=100)

                linear_classifier = LinearSVC(dual=False).fit(X, Y)
                scores[(i, j, k)] = X, Y, linear_classifier.score(X, Y)

    for i, j, k in sorted(scores, key=lambda x: scores[x][2], reverse=True)[:10]:
        X, Y, score = scores[(i, j, k)]

        plt.scatter(X[~Y][:, 0], X[~Y][:, 1], label='Label = 0')
        plt.scatter(X[Y][:, 0], X[Y][:, 1], label='Label = 1')
        plt.legend()
        plt.xlabel(f'{i}')
        plt.ylabel(f'{j}')
        plt.title(f'Label {label_names[k]} \nIndices {i} {j} Score {score}')
        plt.show()


# viz 3#
def project_2d(latents, labels, label_names):
    latent_dataset = TensorDataset(latents)
    latent_dataloader = DataLoader(latent_dataset, batch_size=64)
    latent_autoencoder = Autoencoder(input_dim=latents.shape[1], hidden_dim=2)
    latent_autoencoder.train_model(latent_dataloader, 100)

    projection = []
    for [sample] in latent_dataloader:
        projection.append(latent_autoencoder.encode(sample))
    projection = torch.cat(projection)

    with torch.no_grad():
        _, axes = plt.subplots(ncols=8, nrows=5, figsize=(32, 20))
        for i, axis in tqdm(enumerate(axes.flatten()), 'Axes', total=40):
            label = labels[:, i]
            axis.scatter(projection[label][:, 0], projection[label][:, 1], label='label=1', alpha=0.5)
            axis.scatter(projection[~label][:, 0], projection[~label][:, 1], label='label=0', alpha=0.5)
            axis.set_title(f'Label {i + 1}: {label_names[i]}')
            axis.legend()
        plt.show()


@torch.no_grad()
def show_label_variance(latents):
    plt.figure(figsize=(12, 5))
    plt.bar(range(latents.shape[1]), latents.var(dim=0))
    plt.xticks(range(latents.shape[1]))
    plt.show()


@torch.no_grad()
def play_with_friends(model, image_file, manipulations):
    name = image_file.split('.')[0].split('/')[-1]
    model = model.cpu()
    resize = Resize((64, 64))
    to_tensor = ToTensor()

    image = to_tensor(resize(Image.open(image_file)))
    plot_image(image, title='Resized Image')

    reconstruction = model(image.unsqueeze(0)).squeeze()
    plot_image(reconstruction, title='Reconstructed Image')

    latent_o = model.get_latent(image.unsqueeze(0))
    discrete_index = [i for i, val in enumerate(latent_o[:, 32:].squeeze()) if val == 1][0]
    for i in manipulations:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        min, max = manipulations[i]
        latent = torch.clone(latent_o)

        fig.suptitle(f'Dimension {i} Discrete variable {discrete_index+1}')
        latent[0, i] = min
        plot_image(model.decode(latent).squeeze(), title='Min', axe=ax1)

        latent[0, i] = max
        plot_image(model.decode(latent).squeeze(), title='Max', axe=ax2)
        plt.savefig(f'viz/{name}_{i}')
        plt.show()


def plot_image(image, title=None, axe=None):
    if axe is None:
        if title is not None:
            plt.title(title)
        plt.imshow(image.permute((1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        if title is not None:
            axe.set_title(title)
        axe.imshow(image.permute((1, 2, 0)))
        axe.set_xticks([])
        axe.set_yticks([])


def main():
    train_dataset = CelebADataset('celeba_resized', limit=5000)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    checkpoint = torch.load('logs/full-150epochs/checkpoint_149.pt')
    model = JointVAE(latent_spec={'cont': 32, 'disc': [10]}).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        latents = []
        labels = []
        for batch, batch_labels in tqdm(train_loader, desc='Encoding Dataset'):
            latents.append(model.get_latent(batch.cuda()).cpu())
            labels.append(batch_labels)

        latents = torch.cat(latents)
        labels = torch.cat(labels)

    varied_dims = (latents.var(dim=0) > latents.var(dim=0).mean())[:32]
    manipulations = {}
    for dim, val in enumerate(varied_dims):
        if val:
            manipulations[dim] = (min(latents[:, dim]), max(latents[:, dim]))
    play_with_friends(model, 'our_images/hamir_tazan.jpeg', manipulations)

    # show_label_variance(latents)
    # label_names = utils.get_label_names()
    # varied_latents = latents[:, latents.var(dim=0) > latents.var(dim=0).mean()]
    # project_2d(varied_latents, labels, label_names)
    # find_correlated_dimensions(latents[:, :model.latent_cont_dim], labels, label_names)


if __name__ == '__main__':
    main()
