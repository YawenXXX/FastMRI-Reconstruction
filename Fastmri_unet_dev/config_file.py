
"Some global configure settings here"
import os
ROOT_DIR = os.path.abspath('.')

INPUT_DATA_DIR = os.path.join(os.path.join(ROOT_DIR, 'dataset'), 'singlecoil_train')
INPUT_VALID_DATA_DIR = os.path.join(os.path.join(ROOT_DIR, 'dataset'), 'singlecoil_val')
INPUT_TEST_DIR = os.path.join(os.path.join(ROOT_DIR, 'dataset'), 'singlecoil_val')
INPUT_ANOTATION_DIR = ""
OUTPUT_DIR = os.path.join(ROOT_DIR, "saved_models_slice3_dp0.0_win11")
LOG_FILE_PATH = os.path.join(ROOT_DIR, "val_loss_log_slice3_dp0.0_win11")
LOG_FILE_TRAIN = os.path.join(ROOT_DIR, "train_loss_log_slice3_dp0.0_win11")
RECONSTRUCTION_DIR = os.path.join(ROOT_DIR, "reconstructions")

TRAIN_RATIO = 1  # ow many percentage of data we are using to train"
BATCH_SIZE = 4
SLICES = [13, 16, 19, 22, 25]
CROP_SIZE = (20, 20)
MAX_FILE_LIMIT = 1000 # how many files we want to process
ACCELERATE_RATE = 4
NUM_CHANNEL_FIRST_LAYER_OUTPUT = 32 # number of output channels of the first layer of the model
NUM_POOL_LAYERS = 4
DROPOUT_PROB = 0.0