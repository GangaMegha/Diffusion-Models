from logger import setup_logger
import logging
import gc
import os
import sys

import pandas as pd
import numpy as np

from config import LEVEL, TRAIN, MODEL, MODEL_TYPE, DATASET_NAME, CONFIG, CHECKPOINT_PATH, RESULT_PATH, LOG_PATH
from train.trainer import Trainer
from data_loader import load_data



if MODEL_TYPE == MODEL.UNET:
    from models.unet import Unet as Model


# Setup logger
logger = logging.getLogger('main')
logger = setup_logger(logger, '', '', '%(levelname)s | %(name)s | %(message)s', LEVEL.value)

def main():
    # Load model config
    cfg = CONFIG[MODEL_TYPE.value][DATASET_NAME]

    # Extract data
    logger.debug("Loading Data")
    train_dataloader = load_data(DATASET_NAME, phase="train", grayscale=cfg["grayscale"], shuffle=True)
    test_dataloader = load_data(DATASET_NAME, phase="test", grayscale=cfg["grayscale"], shuffle=False)
    logger.debug("Successfully Loaded Data")

    if TRAIN:
        logger.debug("Loading Model")
        # Create Model
        model = Model(cfg)
        logger.debug("Successfully Loaded Model")

        logger.debug("Creating Trainer")
        # Init the Trainer class
        runner = Trainer(
            model=model,
            train_cfg=cfg,
            model_name=MODEL_TYPE.value,
            dataset_name=DATASET_NAME,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader
        )
        logger.debug("Successfully Created Trainer")

        logger.debug("Starting Training")
        # Start Training Process
        train_log = runner.train()
        logger.debug("Finished Training")

        logger.debug("Saving Training Stats")
        # Logging Training Losses
        train_log = pd.DataFrame(pd.DataFrame(np.array(train_log).T, columns=["train_loss", "val_loss", "FID", "IS"]))
        train_log.to_csv(
            os.path.join(LOG_PATH, f'{DATASET_NAME}/{MODEL_TYPE.value}_train_log.csv')
            )

        # Delete models and use garbage collection to clear memory
        del runner.model
        del model
        gc.collect()

if __name__ == "__main__":

    if  'fashion_mnist' in sys.argv[1]:
        DATASET_NAME = "fashion_mnist"
    elif 'mnist' in sys.argv[1]:
        DATASET_NAME = "mnist"
    elif 'cifar10' in sys.argv[1]:
        DATASET_NAME = "cifar10"
    else:
        sys.exit(0)

    if not os.path.exists(os.path.join(CHECKPOINT_PATH, f'{DATASET_NAME}/')):
        os.makedirs(os.path.join(CHECKPOINT_PATH, f'{DATASET_NAME}/'))

    if not os.path.exists(os.path.join(RESULT_PATH, f'{DATASET_NAME}/')):
        os.makedirs(os.path.join(RESULT_PATH, f'{DATASET_NAME}/'))

    if not os.path.exists(os.path.join(LOG_PATH, f'{DATASET_NAME}/')):
        os.makedirs(os.path.join(LOG_PATH, f'{DATASET_NAME}/'))

    # Call Main
    main()