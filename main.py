"""
You should submit a Zip ( not Rar! ) file containing:
    - Code: as many files as you need (one of them should be “main.py,” which will include the running process)
    - Report: .pdf file (including your names and ids)
    - Saved model: .pkl file (If the file is too big for the Moodle, upload it to your Google-Drive and copy the link
      to your pdf report)
"""
from time import strftime

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import utils
from celeba_dataset import CelebADataset
from modules.joint_vae import JointVAE
from plotting import plot_metrics
from train import train_vae
from utils import log_run

LOG_DIR = "./logs"


def reproduce_hw3():
    """
    This function should be able to reproduce the results that we reported.
    :return:
    """
    hparams = {
        "latent_spec": {'cont': 32, 'disc': [10]},
        "temperature": 0.67,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 150,
        "gamma": 100,
        "C_cont": {"min_val": 0, "max_val": 10, "total_iters": 100000},
        "C_disc": {"min_val": 0, "max_val": 50, "total_iters": 100000},
    }
    prefix = "full"
    model_name = f"{prefix}__{strftime('%Y_%m_%d__%H_%M_%S')}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CelebADataset('celeba_resized')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True
    )

    model = JointVAE(
        latent_spec=hparams["latent_spec"],
        temperature=hparams["temperature"],
    ).to(device)

    num_params = sum([p.numel() for p in model.parameters()])
    print(f"Num params: {num_params}")
    log_run(LOG_DIR, model_name, model, hparams)

    optimizer = Adam(
        params=model.parameters(),
        lr=hparams["lr"],
    )

    path = f"{LOG_DIR}/{model_name}"
    metrics, _ = train_vae(
        model=model,
        log_dir=path,
        dataloader=train_loader,
        num_epochs=hparams["epochs"],
        optimizer=optimizer,
        gamma=hparams["gamma"],
        C_cont=utils.get_capacity_func(**hparams["C_cont"]),
        C_disc=utils.get_capacity_func(**hparams["C_disc"]),
        device=device
    )

    plot_metrics(path, metrics)
