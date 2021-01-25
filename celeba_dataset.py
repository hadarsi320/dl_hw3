import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CelebADataset(Dataset):
    def __init__(self, data_dir='celeba_resized'):
        super(CelebADataset, self).__init__()
        self.data_dir = data_dir
        self.images, self.labels = self.load_images()

    def load_images(self):
        images = []
        labels = []
        for file in tqdm(os.listdir(self.data_dir), desc='Loading Images'):
            image, label = torch.load(f'{self.data_dir}/{file}')
            images.append(image)
            labels.append(label)
        return images, labels

    def __getitem__(self, item):
        return self.images[item], self.labels[item]

    def __len__(self):
        return len(self.images)
