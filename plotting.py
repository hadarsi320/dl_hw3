import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm

import utils
from modules.autoencoder import Autoencoder
from modules.joint_vae import JointVAE

CONT_DIM = 32
VIZ_DIR = 'viz'


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
        print(f'Latent dims ({i}, {j}) Label {label_names[k]} Score {score}')

        plt.scatter(X[~Y][:, 0], X[~Y][:, 1], label='Label = 0', alpha=0.5)
        plt.scatter(X[Y][:, 0], X[Y][:, 1], label='Label = 1', alpha=0.5)
        plt.legend()
        plt.xlabel(f'{i}')
        plt.ylabel(f'{j}')
        plt.title(f'Label {label_names[k]} \nContinuous Latent Dimensions {i}, {j}')
        plt.show()


def find_dim_label_corr(cont_latents, labels, label_names, target_dims):
    scores = {}
    for i in tqdm(target_dims, 'i'):
        for k in range(labels.shape[1]):
            X = cont_latents[:, [i]]
            Y = labels[:, k]
            X, Y = utils.under_sample(X, Y, n=100)

            linear_classifier = LinearSVC(dual=False).fit(X, Y)
            scores[(i, k)] = X, Y, linear_classifier.score(X, Y)

    for index, (i, k) in enumerate(sorted(scores, key=lambda x: scores[x][2], reverse=True)[:10]):
        X, Y, score = scores[(i, k)]
        print(f'{index}. Latent dim {i} Label {label_names[k]} Score {score}')

        plt.vlines(X[~Y][:, 0], -1, 1, colors='blue',
                   label='Label = 0', alpha=0.5)
        plt.vlines(X[Y][:, 0], -1, 1, colors='orange',
                   label='Label = 1', alpha=0.5)
        plt.legend()
        plt.yticks([])
        # plt.xlabel(f'{i}')
        # plt.title(f'Label {label_names[k]} \nContinuous Latent Dimension {i}')
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
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        min, max = manipulations[i]
        middle = (max + min) / 2
        latent = torch.clone(latent_o)

        fig.suptitle(f'Dimension {i} Discrete variable {discrete_index + 1}')
        latent[0, i] = min
        plot_image(model.decode(latent).squeeze(), title='Min', axe=ax1)

        latent[0, i] = middle
        plot_image(model.decode(latent).squeeze(), title='Middle', axe=ax2)

        latent[0, i] = max
        plot_image(model.decode(latent).squeeze(), title='Max', axe=ax3)
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
