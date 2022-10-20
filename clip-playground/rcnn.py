from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.transforms.functional import pil_to_tensor
import torch
from PIL import Image
import numpy as np
import cv2
import clip
import numpy as np

from gather_data import gather_data, LabelMapping

def make_prediction(input_images, input_labels):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    image_list = [pil_to_tensor(Image.fromarray(img))[:3,:,:]/255.  for img in input_images]
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(image_list)
    analyze_i = 3
    imgi_bboxes = predictions[analyze_i]['boxes'].detach().numpy()
    imgi = image_list[analyze_i].numpy().transpose([1, 2, 0])
    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Find largest bbox
    largest_size = 0
    largest_bbox = None
    for bbox in imgi_bboxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        size = width * height
        if size > largest_size:
            largest_size = size
            largest_bbox = tuple(int(i) for i in bbox)


    #for bbox in imgi_bboxes:
    cv2.rectangle(imgi, (
        int(largest_bbox[0]), int(largest_bbox[1])
    ), (
        int(largest_bbox[2]), int(largest_bbox[3])
    ), color, thickness)

    cv2.imshow("Result", imgi)
    cv2.waitKey(0)
    cut_img = imgi[
            largest_bbox[0]:largest_bbox[2], 
            largest_bbox[1]:largest_bbox[3]
    ]
    cut_img = (cut_img*255).astype(np.uint8)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    #image = preprocess(Image.open("flaschentaschen.jpeg")).unsqueeze(0).to(device)
    image = preprocess(Image.fromarray(cut_img)).unsqueeze(0).to(device)
    text = clip.tokenize(input_labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    
    print("Labels: ", input_labels)
    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


if __name__ == "__main__":
    images, raw_labels = gather_data('data')
    mapper = LabelMapping("mappings.yaml", raw_labels)
    image_list = []
    for key in images:
        if key == "noisebridge-poster":
            image_list += images[key]
    make_prediction(image_list, mapper.output_labels)
