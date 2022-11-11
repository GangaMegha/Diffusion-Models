'''
Note : This script is taken from part of https://huggingface.co/blog/annotated-diffusion
'''

import numpy as np
from torchvision import transforms 
from datasets import load_dataset

from torch.utils.data import DataLoader

from config import DATASET


def model_transform():
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    return transform

def reverse_transform():
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    return transform

def transforms_grayscale(data):
   data["pixel_values"] = [model_transform()(image.convert("L")) for image in data["image"]]
   del data["image"]

   return data

def transforms_all(data):
   data["pixel_values"] = [model_transform()(image) for image in data["img"]]
   del data["img"]

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