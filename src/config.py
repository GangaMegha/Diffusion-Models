T = 200

DATASET = {
    "fashion_mnist":{
            "image_size" : 28,
            "channels" : 1,
            "batch_size" : 128
    }
}


class MODEL(Enum):
    UNET = 'Unet' # Implementation from https://huggingface.co/blog/annotated-diffusion


MODEL_TYPE = MODEL.UNET

CHECKPOINT_PATH = "../checkpoint/"

RESULT_PATH = "../results/"

CONFIG = {
    "unet" : {
        "fashion_mnist" : {
            "epochs" : 10,
            "patience" : 5,
            "loss_type" : "huber",
            "lr" : 1e-3,
            "weight_decay" : 0

        }
    }
}
