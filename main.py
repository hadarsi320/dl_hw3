"""
You should submit a Zip ( not Rar! ) file containing:
    - Code: as many files as you need (one of them should be “main.py,” which will include the running process)
    - Report: .pdf file (including your names and ids)
    - Saved model: .pkl file (If the file is too big for the Moodle, upload it to your Google-Drive and copy the link
      to your pdf report)
"""
from time import strftime

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


def main(hparams, train_loader):
    prefix = "full"
    model_name = f"{prefix}__{strftime('%Y_%m_%d__%H_%M_%S')}"
    device = "cuda"

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


if __name__ == '__main__':
    hparams = {
        "batch_size": 64,
    }
    train_dataset = CelebADataset('celeba_resized')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True
    )

    # run 1: paper hyper parameters with different batch sizes
    hparams = {
        "latent_spec": {'cont': 32, 'disc': [10]},
        "temperature": 0.67,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 100,
        "gamma": 100,
        "C_cont": {"min_val": 0, "max_val": 10, "total_iters": 100000},
        "C_disc": {"min_val": 0, "max_val": 50, "total_iters": 100000},
    }
    main(hparams, train_loader)

    # run 2: check how temp affects, low epochs so we don't waste time
    for temp in [0.1, 0.33, 1, 5]:
        hparams = {
            "latent_spec": {'cont': 32, 'disc': [10]},
            "temperature": temp,
            "batch_size": 64,
            "lr": 5e-4,
            "epochs": 30,
            "gamma": 100,
            "C_cont": {"min_val": 0, "max_val": 10, "total_iters": 100000},
            "C_disc": {"min_val": 0, "max_val": 50, "total_iters": 100000},
        }
        main(hparams, train_loader)

    # run 3: check how gamma affects, low epochs so we don't waste time
    for gamma in [1, 10, 100, 1000]:
        hparams = {
            "latent_spec": {'cont': 32, 'disc': [10]},
            "temperature": 0.67,
            "batch_size": 64,
            "lr": 5e-4,
            "epochs": 30,
            "gamma": gamma,
            "C_cont": {"min_val": 0, "max_val": 10, "total_iters": 100000},
            "C_disc": {"min_val": 0, "max_val": 50, "total_iters": 100000},
        }
        main(hparams, train_loader)

    # run 4: check how latent_spec affects, VERY low epochs so we don't waste time
    for cont in [1, 2, 4, 32, 64]:
        for disc in [[10], [2], [2] * 40]:
            hparams = {
                "latent_spec": {'cont': cont, 'disc': disc},
                "temperature": 0.67,
                "batch_size": 64,
                "lr": 5e-4,
                "epochs": 10,
                "gamma": 100,
                "C_cont": {"min_val": 0, "max_val": 10, "total_iters": 100000},
                "C_disc": {"min_val": 0, "max_val": 50, "total_iters": 100000},
            }
            main(hparams, train_loader)

    # run 5: check how C_cont affects, low epochs so we don't waste time
    for max_Cc in [1, 10, 100, 1000]:
        hparams = {
            "latent_spec": {'cont': 32, 'disc': [10]},
            "temperature": 0.67,
            "batch_size": 64,
            "lr": 5e-4,
            "epochs": 30,
            "gamma": 100,
            "C_cont": {"min_val": 0, "max_val": max_Cc, "total_iters": 100000},
            "C_disc": {"min_val": 0, "max_val": 50, "total_iters": 100000},
        }
        main(hparams, train_loader)
