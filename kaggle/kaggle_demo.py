# Imports
# The notebooks is self-contained
# It has very few imports
# No external dependencies (only the model weights)
# No train - inference notebooks
# We only rely on Pytorch
import os
import time
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 

import torch
import torchvision
from torchvision.transforms import ToPILImage
from torch.nn import DataParallel 
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

from load_data import CellDataset, CellTestDataset, get_transform
from get_model_fn import get_model
#### ------------------------------------------------------------------------------------------------

#### SETUP


INPUT_DIR = "/home/culteejen/development/kaggle-sartorius"
TRAIN_CSV = os.path.join(INPUT_DIR, "sartorius-cell-instance-segmentation/train.csv")
TRAIN_PATH = os.path.join(INPUT_DIR, "sartorius-cell-instance-segmentation/train")
TEST_PATH = os.path.join(INPUT_DIR, "sartorius-cell-instance-segmentation/test")

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

# Use a StepLR scheduler if True. Not tried yet.
USE_SCHEDULER = False

# Amount of epochs
NUM_EPOCHS = 8




MIN_SCORE = 0.59

# For the eval part in Tensorboard ============
WIDTH = 704
HEIGHT = 520

# Changes the confidence required for a pixel to be kept for a mask. 
# Only used 0.5 till now.
MASK_THRESHOLD = 0.5
# =============================

df_train = pd.read_csv(TRAIN_CSV, nrows=5000 if TEST else None)
ds_train = CellDataset(TRAIN_PATH, df_train, resize=False, transforms=get_transform(train=True))
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, 
                      num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
ds_test = CellTestDataset(TEST_PATH, transforms=get_transform(train=False))
dl_test = DataLoader(ds_test, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


model = get_model()
for param in model.parameters():
    param.requires_grad = True
    
model.train();

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

parallel_net = DataParallel(model, device_ids = [0,1,2])

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

n_batches = len(dl_train)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Starting epoch {epoch} of {NUM_EPOCHS}")
    
    time_start = time.time()
    loss_accum = 0.0
    loss_mask_accum = 0.0

    for img_num, (img_tuple, image_id_tuple) in enumerate(dl_test, 1):
        img = img_tuple[0]
        # Need to index because DataLoader iterates by tuples rather than individually (to support batches).
        writer.add_image('test_input_images', img.to(DEVICE), img_num)
        model.eval()
        with torch.no_grad():
            preds = model([img.to(DEVICE)])[0]
        # Don't think we need any of this, as we're writing a tensor. Keeping in case I'm wrong.
        # all_preds_masks = np.zeros((HEIGHT, WIDTH))
        # for mask in preds['masks'].cpu().detach().numpy():
        #     all_preds_masks = np.logical_or(all_preds_masks, mask[0] > MASK_THRESHOLD)

        all_detections = torch.zeros((1, HEIGHT, WIDTH))
        for mask in preds['masks'].cpu().detach():
            all_detections = torch.logical_or(all_detections, mask > MASK_THRESHOLD)
        writer.add_image('test_prediction', all_detections, img_num)
        model.train()
    
    for batch_idx, (images, targets) in enumerate(dl_train, 1):
    
        # Predict
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # loss_dict = model(images, targets)
        loss_dict = parallel_net(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        loss_mask = loss_dict['loss_mask'].item()
        loss_accum += loss.item()
        loss_mask_accum += loss_mask

        # TODO : Add a sample prediction in Tensorboard.
        # writer.add_graph(model, images)
        current_index = (epoch-1)*n_batches + batch_idx
        writer.add_scalar('Train_loss', loss.item(), current_index)
        writer.add_scalar('Train_mask_only_loss', loss_mask, current_index)
        
        if batch_idx % 10 == 1:
            print(f"    [Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f}")
    

    if USE_SCHEDULER:
        lr_scheduler.step()
    
    # Train losses
    train_loss = loss_accum / n_batches
    train_loss_mask = loss_mask_accum / n_batches
    
    
    elapsed = time.time() - time_start

    
    
    torch.save(model.state_dict(), f"pytorch_model-e{epoch}.bin")
    prefix = f"[Epoch {epoch:2d} / {NUM_EPOCHS:2d}]"
    print(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}")
    print(f"{prefix} Train loss: {train_loss:7.3f}. [{elapsed:.0f} secs]")
#### ------------------------------------------------------------------------------------------------

writer.close()


