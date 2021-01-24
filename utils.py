import os

import torch

from main import LOG_DIR


def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    if torch.cuda.is_available():
        U = U.cuda()
    return - torch.log(eps - torch.log(U + eps))


def get_capacity_func(min_val, max_val, total_iters):
    return lambda i: min(max_val,
                         ((max_val - min_val) * i / float(total_iters) + min_val))


def log_run(model_name, model, hparams):
    path = f"{LOG_DIR}/{model_name}"
    os.mkdir(path)
    with open(f"{path}/description.txt", "w") as f:
        hparams_string = "\n".join([f"\t{k:30} {v}" for k, v in hparams.items()])
        f.writelines([model_name, "\n\n", str(model), "\n\n", "hparams:\n", hparams_string])