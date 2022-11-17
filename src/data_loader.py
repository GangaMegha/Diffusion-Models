'''
Note : This script is taken from part of https://huggingface.co/blog/annotated-diffusion
        and modified by Ganga Meghanath
'''

import numpy as np
import torch
from torchvision import transforms 
from datasets import load_dataset

from torch.utils.data import DataLoader

from config import DATASET


def model_transform():
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            # transforms.CenterCrop(128), # CelebA
            # transforms.Resize(64),
            transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    return transform

def reverse_transform():
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1)  * 0.5),
        transforms.Lambda(lambda t: t * 255),
        transforms.Lambda(lambda t: t.to(dtype=torch.uint8))
    ])
    return transform

def transforms_grayscale(data):
   data["pixel_values"] = [model_transform()(image.convert("L")) for image in data["image"]]
   del data["image"]

   return data

def transforms_all(data):
   data["pixel_values"] = [model_transform()(image) for image in data["img"]]
   del data["img"] # CelebA : image

   return data

def load_data(dataset_name="fashion_mnist", phase="train", grayscale=True, shuffle=True):
    # load dataset from the hub
    dataset = load_dataset(dataset_name)

    if grayscale:
        transformed_dataset = dataset.with_transform(transforms_grayscale).remove_columns("label")
    else:
        transformed_dataset = dataset.with_transform(transforms_all).remove_columns("label")

    # create dataloader
    dataloader = DataLoader(transformed_dataset[phase], batch_size=DATASET[dataset_name]["batch_size"], shuffle=shuffle)

    return dataloader