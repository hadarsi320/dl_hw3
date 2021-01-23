import torch
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

import matplotlib.pyplot as plt

from modules.joint_vae import JointVAE

from tqdm import tqdm

BATCH_SIZE = 64
TEMP = 0.67
EPOCHS = 100
GAMMA = 100
C_cont = (0, 50, 1e5)
C_disc = (0, 10, 1e5)
LEARNING_RATE = 5e-4


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = Compose([ToTensor()])
    train_dataset = dsets.CelebA(root='/datashare/',
                                 split='train', transform=transforms,
                                 download=False)  # make sure you set it to False

    # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    resize = Resize((64, 64))

    # latent_space = {'cont': 32, 'disc': [10]}
    latent_space = {'disc': [10]}
    model = JointVAE(latent_space, TEMP, device)
    model.to(device)

    for i, (x, _) in tqdm(enumerate(train_dataset), total=100):
        if i == 100:
            break
        x = resize(x).unsqueeze(0).to(device)
        rec = model(x)


if __name__ == '__main__':
    main()
