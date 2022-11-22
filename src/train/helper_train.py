'''
Note : This script is taken from part of https://huggingface.co/blog/annotated-diffusion
        and modified by Ganga Meghanath.
        Added l2 loss with weight from original variance diffusion paper -> clipped the weights to stabilize training
'''

import torch
import torch.nn.functional as F

from diffusion.forward_diffusion import q_sample, extract


def p_losses(denoise_model, x_start, t, variance_dict, loss_type="l1", noise=None, clip=1):
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
    elif loss_type == "l2_weighted":
        beta_t = extract(variance_dict["betas"], t, x_noisy.shape)
        sigma_t_2 = extract(variance_dict["posterior_variance"], t, x_noisy.shape)
        weight = (beta_t**2)/(2 * sigma_t_2 * (1-beta_t) * (1-extract(variance_dict["alphas_cumprod"], t, x_noisy.shape)))
        # If weights are not clipped, training is unstable and loss goes to nan 
        weight = torch.clip(weight, 0.0, clip) # 1, 10, 100
        loss = torch.mean(weight * (noise - predicted_noise) ** 2)

        # weight = 2*torch.log(beta_t) - torch.log(2 * sigma_t_2) - torch.log(1-beta_t) - torch.log(1-extract(variance_dict["alphas_cumprod"], t, x_noisy.shape))
        # weight = torch.clip(weight, 1.0, 1.0)
        # loss = torch.mean( torch.exp(0 + 2*torch.log(torch.abs(noise - predicted_noise))) )
        
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