import torch
import torch.nn.functional as F


def mask_to_token_loss_weight(mask, latent_size=32, lambda_edit=3.0, lambda_bg=1.0):
    if mask.ndim == 4 and mask.shape[-1] in (1, 3):
        mask = mask.permute(0, 3, 1, 2)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    if mask.shape[1] > 1:
        mask = mask[:, :1]

    mask = mask.float()
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask = (mask > 0.5).float()
    mask = F.interpolate(mask, size=(latent_size, latent_size), mode="nearest")
    mask = mask.flatten(1)
    return mask * float(lambda_edit) + (1.0 - mask) * float(lambda_bg)
