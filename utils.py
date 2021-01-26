import os

import numpy as np
import torch


def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    if torch.cuda.is_available():
        U = U.cuda()
    return - torch.log(eps - torch.log(U + eps))


def get_capacity_func(min_val, max_val, total_iters):
    return lambda i: min(max_val,
                         ((max_val - min_val) * i / float(total_iters) + min_val))


def log_run(log_dir, model_name, model, hparams):
    path = f"{log_dir}/{model_name}"
    os.mkdir(path)
    with open(f"{path}/description.txt", "w") as f:
        hparams_string = "\n".join([f"\t{k:30} {v}" for k, v in hparams.items()])
        f.writelines([model_name, "\n\n", str(model), "\n\n", "hparams:\n", hparams_string])


def get_label_names(file='/datashare/celeba/list_attr_celeba.txt'):
    line = open(file).readlines()[1]
    return np.array(line.split())


def under_sample(data, labels, n=100):
    data_0 = data[~labels][:n]
    data_1 = data[labels][:n]
    if len(data_0) < n or len(data_1) < n:
        return None
    new_labels = torch.zeros(2 * n, dtype=torch.bool)
    new_labels[n:] = 1
    return torch.cat([data_0, data_1]), new_labels
