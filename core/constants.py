"""
Manages all the constants used across the project.
"""

DATASET_FOLDER_PATH = '../../data/'
ROUGH_TRAIN_DATASET_PATH = DATASET_FOLDER_PATH + 'rough_train_dataset.csv'
ROUGH_VAL_DATASET_PATH = DATASET_FOLDER_PATH + 'rough_val_dataset.csv'
CLEAN_TRAIN_DATASET_PATH = DATASET_FOLDER_PATH + 'clean_train_dataset.csv'
CLEAN_VAL_DATASET_PATH = DATASET_FOLDER_PATH + 'clean_val_dataset.csv'
TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-5
MODEL_FILE_PATH = DATASET_FOLDER_PATH + 'model.bin'
OUTPUT_ANSWER_FILE_PATH = DATASET_FOLDER_PATH + 'data/answer.txt'
