'''
Note : This script is taken from part of https://huggingface.co/blog/annotated-diffusion
'''

import torch
import torch.nn.functional as F

from diffusion.forward_diffusion import q_sample


def p_losses(denoise_model, x_start, t, variance_dict, loss_type="l1", noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, variance_dict["sqrt_alphas_cumprod"], variance_dict["sqrt_one_minus_alphas_cumprod"], noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr