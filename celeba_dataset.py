import os

import torch
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, data_dir):
        super(CelebADataset, self).__init__()
        self.data_dir = data_dir
        self.images = self.load_images()

    def load_images(self):
        images = []
        for file in os.listdir(self.data_dir):
            images.append(torch.load(f'{self.data_dir}/{file}'))
        return images

    def __getitem__(self, item):
        return self.images[item]

    def __len__(self):
        return len(self.images)
