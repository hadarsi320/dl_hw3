import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    if torch.cuda.is_available():
        U = U.cuda()
    return - torch.log(eps - torch.log(U + eps))
