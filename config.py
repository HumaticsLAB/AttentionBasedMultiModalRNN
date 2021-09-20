import torch

DEVICE = torch.device('cuda:0')
DATASET_PATH = "dataset/images"
TRAIN_DATASET = "dataset/train.csv"
TEST_DATASET = "dataset/test.csv"
COMPOSED_GTREND = "dataset/gtrends.csv"
CATEG_DICT = "category_labels.pt"
COLOR_DICT = "color_labels.pt"
FAB_DICT = "fabric_labels.pt"
NUM_EPOCHS = 50
USE_TEACHERFORCING = True
TF_RATE = 0.5
LEARNING_RATE = 0.0001
NORMALIZATION_VALUES_PATH = "dataset/normalization_scale.npy"
BATCH_SIZE= 128
SHOW_PLOTS = False
NUM_WORKERS = 8
USE_EXOG = True
EXOG_NUM = 3
EXOG_LEN = 52
HIDDEN_SIZE = 300
SAVED_FEATURES_PATH = "incv3_features"
USE_SAVED_FEATURES = False
NORM = False
model_types = ["image", "concat", "residual", "cross"]
MODEL = 1
