import torch
import torch.nn.functional as F


def make_negative_permutation(batch_size, device):
    if batch_size < 2:
        return None
    perm = torch.randperm(batch_size, device=device)
    for _ in range(8):
        if not torch.any(perm == torch.arange(batch_size, device=device)):
            return perm
        perm = torch.randperm(batch_size, device=device)
    return torch.roll(torch.arange(batch_size, device=device), shifts=1)


def margin_contrastive_loss(loss_correct, loss_wrong, margin):
    return F.relu(margin + loss_correct - loss_wrong)
