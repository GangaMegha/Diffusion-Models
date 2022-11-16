from enum import Enum
import logging

CHECKPOINT_PATH = "./checkpoint/"
RESULT_PATH = "./results/"
LOG_PATH = f'./log'

DATASET = {
    "fashion_mnist":{
            "image_size" : 28,
            "channels" : 1,
            "batch_size" : 128
    },
    "mnist":{
            "image_size" : 28,
            "channels" : 1,
            "batch_size" : 128
    },
    "cifar10":{
            "image_size" : 32,
            "channels" : 3,
            "batch_size" : 128
    }
    
}
DATASET_NAME = "fashion_mnist"


class MODEL(Enum):
    UNET = 'Unet' # Implementation from https://huggingface.co/blog/annotated-diffusion


MODEL_TYPE = MODEL.UNET
TRAIN = True


class LOGGING(Enum):
    DEBUG = logging.DEBUG # All messages will be displayed
    INFO = logging.INFO # Only the training/eval loss and results will be displayed 
    WAN = logging.WARN # Nothing will be displayed

LEVEL = LOGGING.DEBUG

CONFIG = {
    "Unet" : {
        "fashion_mnist" : {
            "image_size" : 28,
            "channels" : 1,
            "grayscale" : True,
            "batch_size" : 128,
            "T" : 500,
            "var_schedule" : "linear", # Choose from cosine, linear, quadratic, sigmoid
            "epochs" : 200,
            "patience" : 10,
            "loss_type" : "huber",
            "lr" : 1e-3,
            "weight_decay" : 0,
            "dim_mults" : (1, 2, 4,),
        },

        "mnist" : {
            "image_size" : 28,
            "channels" : 1,
            "grayscale" : True,
            "batch_size" : 128,
            "T" : 250,
            "var_schedule" : "linear", # Choose from cosine, linear, quadratic, sigmoid
            "epochs" : 100,
            "patience" : 5,
            "loss_type" : "huber",
            "lr" : 1e-3,
            "weight_decay" : 0,
            "dim_mults" : (1, 2, 4,),
        },
        "cifar10" : {
            "image_size" : 32,
            "channels" : 3,
            "grayscale" : False,
            "batch_size" : 128,
            "T" : 1000,
            "var_schedule" : "linear", # Choose from cosine, linear, quadratic, sigmoid
            "epochs" : 500,
            "patience" : 10,
            "loss_type" : "huber",
            "lr" : 1e-3,
            "weight_decay" : 0,
            "dim_mults" : (1, 2, 4, 8,),
        }
    }
}
