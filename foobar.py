import os
import sys
from ctypes import sizeof

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
    pass


if __name__ == '__main__':
    main()
