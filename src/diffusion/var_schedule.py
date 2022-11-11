'''
Note :  This script is initially taken from part of https://huggingface.co/blog/annotated-diffusion
        and modified by Ganga Meghanath.
'''

import torch

def cosine_beta_schedule(T, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)
    ft = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = ft / ft[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(T):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, T)

def quadratic_beta_schedule(T):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, T) ** 2

def sigmoid_beta_schedule(T):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, T)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def alpha_beta(T, schedule="linear"):
    # define beta schedule
    if schedule=="cosine":
        betas = cosine_beta_schedule(T)
    elif schedule=="linear":
        betas = linear_beta_schedule(T)
    elif schedule=="quadratic":
        betas = quadratic_beta_schedule(T)
    elif schedule=="sigmoid":
        betas = sigmoid_beta_schedule(T)
    else:
        print("\n\n\n\t\tVariance Schedule is UNKNOWN!! \n\nPlease choose schedule from one of the following:  \n\tcosine, \n\tlinear, \n\tquadratic, \n\tsigmoid")
        raise NotImplementedError()

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    variance_dict = {
        "betas" : betas,
        "alphas_cumprod" : alphas_cumprod,
        "sqrt_recip_alphas" : sqrt_recip_alphas,
        "sqrt_alphas_cumprod" : sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod" : sqrt_one_minus_alphas_cumprod,
        "posterior_variance" : posterior_variance,
    }

    return variance_dict