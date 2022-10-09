from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.transforms.functional import pil_to_tensor
import torch
from images import images
from PIL import Image
import numpy as np
import cv2

model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
image_list = [pil_to_tensor(Image.open(img))[:3,:,:]/255.  for img in images]
for img in image_list:
    print(img.shape)
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
print(x[0].shape)
predictions = model(image_list)
analyze_i = 3
imgi_bboxes = predictions[analyze_i]['boxes'].detach().numpy()
imgi = image_list[analyze_i].numpy().transpose([1, 2, 0])
print(imgi.shape)
# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

for bbox in imgi_bboxes:
    print(bbox)
    cv2.rectangle(imgi, (
        int(bbox[0]), int(bbox[1])
    ), (
        int(bbox[2]), int(bbox[3])
    ), color, thickness)

cv2.imshow("Result", imgi)
cv2.waitKey(0)


#predictions = model(x)
