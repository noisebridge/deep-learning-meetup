import torch
import torchvision.models.detection

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import numpy as np

# Normalize to resnet mean and std if True.
NORMALIZE = False

BOX_DETECTIONS_PER_IMG = 539

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Train loop

# Get the Mask R-CNN model
# The model does classification, bounding boxes and MASKs for individuals, all at the same time
# We only care about MASKS
def get_model():
    # This is just a dummy value for the classification head
    NUM_CLASSES = 2

    if NORMALIZE:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   box_detections_per_img=BOX_DETECTIONS_PER_IMG,
                                                                   image_mean=RESNET_MEAN,
                                                                   image_std=RESNET_STD)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                  box_detections_per_img=BOX_DETECTIONS_PER_IMG)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)

    # Place it here to consolidate
    model.to(DEVICE)
    return model

def load_from_file(model, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)


def compute_iou(image_mask, target_mask, threshold):
    image_thresholded = (image_mask > threshold).long()
    intersection = image_mask * target_mask
    union = torch.minimum(image_thresholded + target_mask, torch.tensor(1.0))
    total_intersection = intersection.sum()
    total_union = union.sum()
    if total_union == 0:
        return tensor(float('nan'))
    else:
        return total_intersection / total_union

def compute_map_iou(image_mask, target_mask):
    sum_iou = torch.tensor(0)
    min_thresh = 0.5
    max_thresh = 1.0
    step = 0.05
    n_steps = (max_thresh - min_thresh) / step
    for thresh in np.arange(min_thresh, max_thresh, step):
        iou = compute_iou(image_mask, target_mask, thresh)
        sum_iou = sum_iou + iou
    avg_iou = sum_iou / n_steps
    return avg_iou


if __name__ == "__main__":
    # Override pythorch checkpoint with an "offline" version of the file
    # !mkdir -p /root/.cache/torch/hub/checkpoints/
    # !cp ../input/cocopre/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth /root/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
    model = get_model()


