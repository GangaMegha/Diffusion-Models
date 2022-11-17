'''
Note : This script is taken from part of https://huggingface.co/blog/annotated-diffusion
        and modified by Ganga Meghanath
        Added code for sigma_t = beta_t or posterior variance
'''

import torch
from tqdm.auto import tqdm

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, t_index, variance_dict, sigma_beta_t=False):
    betas_t = extract(variance_dict["betas"], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        variance_dict["sqrt_one_minus_alphas_cumprod"], t, x.shape
    )
    sqrt_recip_alphas_t = extract(variance_dict["sqrt_recip_alphas"], t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        if sigma_beta_t:
            posterior_variance_t = extract(variance_dict["betas"], t, x.shape)
        else:
            posterior_variance_t = extract(variance_dict["posterior_variance"], t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, variance_dict, shape, T):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in tqdm(reversed(range(0, T)), desc='sampling loop time step', total=T):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, variance_dict)
        if i%50==0:
            imgs.append(img.cpu())
    imgs.append(img.cpu())
    
    return imgs

@torch.no_grad()
def sample(model, variance_dict, cfg, sample_cnt=16):
    return p_sample_loop(model, variance_dict, shape=(sample_cnt, cfg.get('channels'), cfg.get('image_size'), cfg.get('image_size')), T=cfg.get('T'))