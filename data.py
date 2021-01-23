import torch
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

import matplotlib.pyplot as plt

from modules.joint_vae import JointVAE

from tqdm import tqdm


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = Compose([ToTensor(), Resize((64, 64))])
    train_dataset = dsets.CelebA(root='/datashare/',
                                 split='train', transform=transforms,
                                 download=False)  # make sure you set it to False

    # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for x, _ in tqdm(train_dataset):
        pass


if __name__ == '__main__':
    main()
