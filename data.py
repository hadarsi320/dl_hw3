import os

import torch
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor, Compose, Resize
from tqdm import tqdm

BATCH_SIZE = 64


def resize_data():
    data_dir = 'celeba_resized'
    transforms = Compose([ToTensor(), Resize((64, 64))])
    train_dataset = dsets.CelebA(root='/datashare/',
                                 split='train', transform=transforms,
                                 download=False)  # make sure you set it to False

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for i, (x, y) in tqdm(enumerate(train_dataset), total=len(train_dataset), desc='Saving Images'):
        torch.save((x, y.bool()), '{}/{}.pt'.format(data_dir, str(i).zfill(6)))
