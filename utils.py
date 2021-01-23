import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    if torch.cuda.is_available():
        U = U.cuda()
    return - torch.log(eps - torch.log(U + eps))


def get_capacity_func(min_val, max_val, total_iters):
    return lambda i: min(max_val,
                         ((max_val - min_val) * i / float(total_iters) + min_val))
