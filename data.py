import torch
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

import matplotlib.pyplot as plt

from tqdm import tqdm

BATCH_SIZE = 64


def main():
    transforms = Compose([ToTensor()])
    train_dataset = dsets.CelebA(root='/datashare/',
                                 split='train', transform=transforms,
                                 download=False)  # make sure you set it to False

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    resize = Resize((64, 64))
    for i, (x, y) in tqdm(enumerate(train_dataset)):
        if i == 100:
            break
        x = resize(x).unsqueeze(0)


if __name__ == '__main__':
    main()
