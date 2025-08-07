# fusion/engine.py

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from fusion.utils.visualization import decode_segmap, draw_detection, overlay_binary_mask


def is_box_on_track(bbox, track_mask):
    """
    Check if a detection bounding box intersects with the rail-track segmentation mask.

    Args:
        bbox (list): [x1, y1, x2, y2] bounding box.
        track_mask (ndarray): Binary segmentation mask where rail-track pixels are 1.

    Returns:
        bool: True if box intersects with track mask.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = track_mask.shape

    # Clamp box to image bounds
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)

    roi = track_mask[y1:y2, x1:x2]
    return np.any(roi == 1)


def fuse_single_frame(image, segmentor, detector, config, device):
    """
    Runs segmentation and detection on one image and fuses the result.

    Returns:
        Annotated BGR image
    """
    # Preprocess for segmentation
    input_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Segment
    with torch.no_grad():
        output = segmentor(input_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Extract track mask (binary)
    track_id = config.get("track_class_id", 12)  # default to 12 if unsure
    binary_track_mask = (pred_mask == track_id).astype(np.uint8)

    # Detect
    detection_results = detector(image, verbose=False)[0]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for det in detection_results.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        class_id = int(det.cls[0].item())
        confidence = float(det.conf[0].item())
        label = f"{detector.names[class_id]} {confidence:.2f}"

        on_track = is_box_on_track([x1, y1, x2, y2], binary_track_mask)
        draw_detection(image_bgr, [x1, y1, x2, y2], label, on_track)

    # Overlay track mask
    image_final = overlay_binary_mask(image_bgr, binary_track_mask,
                                            color=(0, 255, 255), alpha=0.5)
    return image_final