import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CelebADataset(Dataset):
    def __init__(self, data_dir='celeba_resized', limit=None):
        super(CelebADataset, self).__init__()
        self.data_dir = data_dir
        self.images, self.labels = self.load_images(limit)

    def load_images(self, limit):
        images = []
        labels = []
        if limit:
            img_list = os.listdir(self.data_dir)[:limit]
        else:
            img_list = os.listdir(self.data_dir)
        for file in tqdm(img_list, desc='Loading Images'):
            image, label = torch.load(f'{self.data_dir}/{file}')
            images.append(image)
            labels.append(label)
        return images, labels

    def __getitem__(self, item):
        return self.images[item], self.labels[item]

    def __len__(self):
        return len(self.images)
