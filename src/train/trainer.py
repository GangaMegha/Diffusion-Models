'''
Note :  This script is taken from part of https://huggingface.co/blog/annotated-diffusion
        and modified by Ganga Meghanath.
'''

from ..config import MODEL, MODEL_TYPE, DATASET

if MODEL_TYPE

import torch
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)