from logger import setup_logger
import logging
import gc
import os

import pandas as pd

from config import LEVEL, TRAIN, MODEL, MODEL_TYPE, DATASET_NAME, CONFIG, CHECKPOINT_PATH, RESULT_PATH, LOG_PATH
from train.trainer import Trainer
from data_loader import load_data



if MODEL_TYPE == MODEL.UNET:
    from models.unet import Unet as Model


# Setup logger
logger = logging.getLogger('main')
logger = setup_logger(logger, '', '', '%(levelname)s | %(name)s | %(message)s', LEVEL.value)

def main():
    # Extract data
    logger.debug("Loading Data")
    grayscale = True if DATASET_NAME in ("fashion_mnist", "mnist") else False            
    train_dataloader = load_data(DATASET_NAME, phase="train", grayscale=grayscale, shuffle=True)
    test_dataloader = load_data(DATASET_NAME, phase="test", grayscale=grayscale, shuffle=False)
    logger.debug("Successfully Loaded Data")

    if TRAIN:
        logger.debug("Loading Model")
        # Load model config
        cfg = CONFIG[MODEL_TYPE.value][DATASET_NAME]
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
        train_log = pd.DataFrame(train_log)
        train_log.to_csv(
            os.path.join(LOG_PATH, f'{DATASET_NAME}/{MODEL_TYPE.value}_train_log.csv')
            )

        # Delete models and use garbage collection to clear memory
        del runner.model
        del model
        gc.collect()

if __name__ == "__main__":

    if not os.path.exists(os.path.join(CHECKPOINT_PATH, f'{DATASET_NAME}/')):
        os.makedirs(os.path.join(CHECKPOINT_PATH, f'{DATASET_NAME}/'))

    if not os.path.exists(os.path.join(RESULT_PATH, f'{DATASET_NAME}/')):
        os.makedirs(os.path.join(RESULT_PATH, f'{DATASET_NAME}/'))

    if not os.path.exists(os.path.join(LOG_PATH, f'{DATASET_NAME}/')):
        os.makedirs(os.path.exists(os.path.join(LOG_PATH, f'{DATASET_NAME}/'))

    # Call Main
    main()