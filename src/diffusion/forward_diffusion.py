'''
Note :  This script is taken from part of https://huggingface.co/blog/annotated-diffusion
        and modified by Ganga Meghanath.
'''

import torch

from data_loader import reverse_transform

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion
def q_sample(x_start, t, var_dict, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(var_dict["sqrt_alphas_cumprod"], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        var_dict["sqrt_one_minus_alphas_cumprod"], t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t, var_dict):
  # add noise
  x_noisy = q_sample(x_start, t=t, var_dict=var_dict)

  # turn back into [0,255] CHW
  noisy_image = reverse_transform()(x_noisy.squeeze())

  return noisy_image