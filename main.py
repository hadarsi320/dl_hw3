"""
You should submit a Zip ( not Rar! ) file containing:
    - Code: as many files as you need (one of them should be “main.py,” which will include the running process)
    - Report: .pdf file (including your names and ids)
    - Saved model: .pkl file (If the file is too big for the Moodle, upload it to your Google-Drive and copy the link
      to your pdf report)
"""
import torchvision.datasets as dsets
from numpy.ma import arange
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

import utils
from celeba_dataset import CelebADataset
from foobar import BATCH_SIZE
from modules.joint_vae import JointVAE
from train_model import train_vae
from torch.optim import Adam


def reproduce_hw3():
    """
    This function should be able to reproduce the results that we reported.
    :return:
    """


if __name__ == '__main__':
    model = JointVAE(
        latent_spec={'cont': 10, 'disc': [3]},
        temperature=0.66,
    ).to('cuda')
    print(f"Num params: {sum([p.numel() for p in model.parameters()])}")
    train_dataset = CelebADataset('celeba_resized')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False  # TODO DEBUG only
    )

    optimizer = Adam(
        params=model.parameters(),
        lr=1e-3,
    )

    train_vae(
        model=model,
        dataloader=train_loader,
        num_epochs=100,
        optimizer=optimizer,
        gamma=100,
        C_cont=utils.get_capacity_func(0, 10, 1e5),
        C_disc=utils.get_capacity_func(0, 50, 1e5),
        device='cuda'
    )
