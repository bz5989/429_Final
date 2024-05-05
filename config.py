# import the necessary packages
import torch
import os
# define path to the original dataset and base path to the dataset
# splits
DATA_PATH = "compressed_dataset"
BASE_PATH = "countries"
BASE_PATH2 = "econs"
BASE_PATH3 = "regions"
# define validation split and paths to separate train and validation
# splits
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "val")
TEST =  os.path.join(BASE_PATH, "test")
TRAIN_ECON = os.path.join(BASE_PATH2, "train")
VAL_ECON = os.path.join(BASE_PATH2, "val")
TEST_ECON =  os.path.join(BASE_PATH2, "test")
TRAIN_REGION = os.path.join(BASE_PATH3, "train")
VAL_REGION = os.path.join(BASE_PATH3, "val")
TEST_REGION =  os.path.join(BASE_PATH3, "test")

# specify mean and standard deviation and image size
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
CAP = 800
ECON_CAP = 8000
REGION_CAP = 1600
REG2 = 0.001
REG2_REGION = 0.01
# determine the device to be used for training and evaluation
DEVICE = "cpu"

# specify training hyperparameters
FEATURE_EXTRACTION_BATCH_SIZE = 256
FINETUNE_BATCH_SIZE = 128
PRED_BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001
LR_FINETUNE = 0.0005

# define paths to store training plots and trained model
FINETUNE_PLOT = os.path.join("output", "finetune.png")
FINETUNE_MODEL = os.path.join("output", "finetune_model.pth")
FINETUNE_PLOT2 = os.path.join("output_econ", "finetune.png")
FINETUNE_MODEL2 = os.path.join("output_econ", "finetune_model.pth")
FINETUNE_PLOT3 = os.path.join("output_region", "finetune.png")
FINETUNE_MODEL3 = os.path.join("output_region", "finetune_model.pth")