"""
You should submit a Zip ( not Rar! ) file containing:
    - Code: as many files as you need (one of them should be “main.py,” which will include the running process)
    - Report: .pdf file (including your names and ids)
    - Saved model: .pkl file (If the file is too big for the Moodle, upload it to your Google-Drive and copy the link
      to your pdf report)
"""
import os
from time import strftime

import torchvision.datasets as dsets
from numpy.ma import arange
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

import utils
from celeba_dataset import CelebADataset
from modules.joint_vae import JointVAE
from train_model_new import train_vae
from torch.optim import Adam


LOG_DIR = "./logs"


def reproduce_hw3():
    """
    This function should be able to reproduce the results that we reported.
    :return:
    """


def log_run(model_name, model, hparams):
    path = f"{LOG_DIR}/{model_name}"
    os.mkdir(path)
    with open(f"{path}/description.txt", "w") as f:
        hparams_string = "\n".join([f"\t{k:30} {v}" for k, v in hparams.items()])
        f.writelines([model_name, "\n\n", str(model), "\n\n", "hparams:\n", hparams_string])


def main():
    prefix = "debug"
    model_name = f"{prefix}__{strftime('%Y_%m_%d__%H_%M_%S')}"
    device = "cuda"
    hparams = {
        "latent_spec": {'cont': 10, 'disc': [3]},
        "temperature": 0.66,
        "batch_size": 128,
        "lr": 5e-4,
        "epochs": 2,
        "gamma": 100,
        "C_cont": {"min_val": 0, "max_val": 10, "total_iters": 1e5},
        "C_disc": {"min_val": 0, "max_val": 50, "total_iters": 1e5},
    }

    model = JointVAE(
        latent_spec=hparams["latent_spec"],
        temperature=hparams["temperature"],
    ).to(device)

    num_params = sum([p.numel() for p in model.parameters()])
    print(f"Num params: {num_params}")
    log_run(model_name, model, hparams)

    train_dataset = CelebADataset('celeba_resized')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True
    )

    optimizer = Adam(
        params=model.parameters(),
        lr=hparams["lr"],
    )

    train_vae(
        model=model,
        model_name=model_name,
        log_dir=f"{LOG_DIR}/{model_name}",
        dataloader=train_loader,
        num_epochs=hparams["epochs"],
        optimizer=optimizer,
        gamma=hparams["gamma"],
        C_cont=utils.get_capacity_func(**hparams["C_cont"]),
        C_disc=utils.get_capacity_func(**hparams["C_disc"]),
        device=device
    )


if __name__ == '__main__':
    main()
