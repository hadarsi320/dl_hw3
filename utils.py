import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    if torch.cuda.is_available():
        U = U.cuda()
    return - torch.log(eps - torch.log(U + eps))


def sample_gumbel_softmax(logits, tau=1, eps=1e-20):
    y = logits + sample_gumbel(logits.size(), eps=eps)
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = sample_gumbel_softmax(logits, tau=tau, eps=eps)

    if hard:
        k = torch.argmax(y_soft, dim=-1)
        y_hard = F.one_hot(k, num_classes=logits.shape[-1])
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y
