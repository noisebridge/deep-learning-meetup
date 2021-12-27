# Imports
# The notebooks is self-contained
# It has very few imports
# No external dependencies (only the model weights)
# No train - inference notebooks
# We only rely on Pytorch
import os
import time

import pandas as pd
# from sklearn.model_selection import train_test_split

import torch
# import torchvision
from torchvision.transforms import ToPILImage
from torch.nn import DataParallel
from torch.utils.data import DataLoader
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

from load_data import CellDataset, CellTestDataset, get_transform, KFoldPyTorch
from get_model_fn import get_model
from get_model_fn import compute_map_iou
from mem_debug import mem_readout
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
USE_SCHEDULER = True

# Amount of epochs
NUM_EPOCHS = 8


MIN_SCORE = 0.59

# For the eval part in Tensorboard ===========
WIDTH = 704
HEIGHT = 520

# Changes the confidence required for a pixel to be kept for a mask.
# Only used 0.5 till now.
MASK_THRESHOLD = 0.5
# =============================

k_fold_pytorch = KFoldPyTorch(n_splits=10, shuffle=True)

df_train = pd.read_csv(TRAIN_CSV, nrows=5000 if TEST else None)
ds_train = CellDataset(TRAIN_PATH, df_train, resize=False, transforms=get_transform(train=True))

for i, train_subsampler, test_subsampler in k_fold_pytorch.splits_iterator(ds_train):
    mem_readout(1)

    if i > 0:
        break

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=train_subsampler,
                          num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    ds_test = CellTestDataset(TEST_PATH, transforms=get_transform(train=False))
    dl_test = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=test_subsampler,
                          num_workers=2, collate_fn=lambda x: tuple(zip(*x)))


    model = get_model()
    for param in model.parameters():
        param.requires_grad = True

    model.train();

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    parallel_net = DataParallel(model, device_ids = [0,1,2])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    n_batches = len(dl_train)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Starting epoch {epoch} of {NUM_EPOCHS}")
        mem_readout(2)

        time_start = time.time()

        loss_accum = 0.0
        loss_mask_accum = 0.0
        batches=0

        model.eval()
        avg_iou = 0
        for batch_idx, (images, targets) in enumerate(dl_test, 1):
            batches+=1

            # Predict
            mem_readout(3)
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            mem_readout(4)
            with torch.no_grad():
                target_masks = []
                for t in targets:
                    image_masks = t['masks']
                    target_masks.append(torch.minimum(torch.sum(image_masks, 0), torch.tensor(1)))
                mem_readout(5)

                preds = parallel_net(images)[0]

            mem_readout(6)
            all_preds_masks = []
            for mask in preds['masks']:
                all_preds_masks.append(torch.maximum(torch.sum(mask, 0), torch.tensor(1)))

            # TODO: Condense the ious into a single score, make sure it looks about right
            mem_readout(7)
            total_iou = 0
            for target_mask, image_mask in zip(target_masks, all_preds_masks):
                total_iou += compute_map_iou(image_mask, target_mask)
            mem_readout(8)
            avg_iou += total_iou.item()
            mem_readout(9)
            del total_iou
            del images
            del targets
            print("IOU: ", avg_iou / batches)
            print("Memory allocated", torch.cuda.memory_allocated())



        avg_iou /= batches
        writer.add_scalar("IOU output", avg_iou, epoch)

        mem_readout(3)
        model.train()
        mem_readout(4)

        # TODO : Add a sample prediction in Tensorboard.
        # writer.add_graph(model, images)
        loss_average = loss_accum / batches
        loss_mask_average = loss_mask_accum / batches
        current_index = epoch
        writer.add_scalar('Test loss', loss_average, current_index)
        writer.add_scalar('Test mask loss', loss_mask_average, current_index)
        print(f"    [Epoc {epoch:2d} / {NUM_EPOCHS:3d}] Batch train loss: {loss_average:7.3f}. Mask-only loss: {loss_mask_average:7.3f}")

        loss_accum = 0.0
        loss_mask_accum = 0.0

        for batch_idx, (images, targets) in enumerate(dl_train, 1):

            mem_readout(5)

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

            del images
            del targets

            if batch_idx % 10 == 1:
                mem_usage_fraction = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                print(f"    [Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f} Memory usage: {mem_usage_fraction:2.3f}")
                mem_readout()

        if USE_SCHEDULER:
            lr_scheduler.step()
            writer.add_scalar("Learning rate", lr_scheduler.get_last_lr()[0], epoch)

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

