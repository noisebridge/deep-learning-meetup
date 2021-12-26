from get_model_fn import get_model, load_from_file
from load_data import CellDataset, get_transform

from matplotlib import pyplot as plt

import numpy as np
import os
import pandas as pd
import torch

INPUT_DIR = "/home/culteejen/development/kaggle-sartorius"
TRAIN_CSV = os.path.join(INPUT_DIR, "sartorius-cell-instance-segmentation/train.csv")
TRAIN_PATH = os.path.join(INPUT_DIR, "sartorius-cell-instance-segmentation/train")
TEST_PATH = os.path.join(INPUT_DIR, "input/sartorius-cell-instance-segmentation/test")

# Reduced the train dataset to 5000 rows
TEST = False

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 2

# No changes tried with the optimizer yet.
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# Normalize to resnet mean and std if True.
NORMALIZE = False

# Use a StepLR scheduler if True. Not tried yet.
USE_SCHEDULER = False

# Amount of epochs
NUM_EPOCHS = 8


BOX_DETECTIONS_PER_IMG = 539


MIN_SCORE = 0.59


WIDTH = 704
HEIGHT = 520

# Changes the confidence required for a pixel to be kept for a mask. 
# Only used 0.5 till now.
MASK_THRESHOLD = 0.5

# Plots: the image, The image + the ground truth mask, The image + the predicted mask
def analyze_train_sample(model, ds_train, sample_index):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')    

    img, targets = ds_train[sample_index]
    ax1.imshow(img.numpy().transpose((1,2,0)))
    ax1.set_title("Image")
    
    masks = np.zeros((HEIGHT, WIDTH))
    for mask in targets['masks']:
        masks = np.logical_or(masks, mask)
    ax2.imshow(img.numpy().transpose((1,2,0)))
    ax2.imshow(masks, alpha=0.3)
    ax2.set_title("Ground truth")
    
    model.eval()
    with torch.no_grad():
        preds = model([img.to(DEVICE)])[0]

    ax3.imshow(img.cpu().numpy().transpose((1,2,0)))
    all_preds_masks = np.zeros((HEIGHT, WIDTH))
    for mask in preds['masks'].cpu().detach().numpy():
        all_preds_masks = np.logical_or(all_preds_masks, mask[0] > MASK_THRESHOLD)
    ax3.imshow(all_preds_masks, alpha=0.4)
    ax3.set_title("Predictions")
    plt.show()


df_train = pd.read_csv(TRAIN_CSV, nrows=5000 if TEST else None)
ds_train = CellDataset(TRAIN_PATH, df_train, resize=False, transforms=get_transform(train=True))

model = get_model()
load_from_file(model, 'pytorch_model-e8.bin')
# NOTE: It puts the model in eval mode!! Revert for re-training
analyze_train_sample(model, ds_train, 20)
