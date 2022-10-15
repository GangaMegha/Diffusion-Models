from datasets import load_dataset

from data_transform import model_transform

from ..config import DATASET

def transforms_grayscale(data):
   data["pixel_values"] = [model_transform(image.convert("L")) for image in data["image"]]
   del data["image"]

   return data

def transforms(data):
   data["pixel_values"] = [model_transform(image) for image in data["image"]]
   del data["image"]

   return data

def load_data(dataset_name="fashion_mnist", grayscale=True):
    # load dataset from the hub
    dataset = load_dataset(dataset_name)

    if grayscale:
        transformed_dataset = dataset.with_transform(transforms_grayscale).remove_columns("label")
    else:
        transformed_dataset = dataset.with_transform(transforms).remove_columns("label")


