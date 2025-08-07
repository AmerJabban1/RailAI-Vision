# depth/utils/visualization.py

import cv2
import numpy as np

# RS19 colormap for 19 classes
RS19_COLORMAP = [
    [128, 64, 128],  # 0: road
    [244, 35, 232],  # 1: sidewalk
    [70, 70, 70],  # 2: construction
    [192, 0, 128],  # 3: tram-track
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],  # 6: traffic-light
    [220, 220, 0],  # 7: traffic-sign
    [107, 142, 35],  # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],  # 10: sky
    [220, 20, 60],  # 11: human
    [230, 150, 140],  # 12: rail-track (optional)
    [0, 0, 142],  # 13: car
    [0, 0, 70],  # 14: truck
    [90, 40, 40],  # 15: trackbed
    [0, 80, 100],  # 16: on-rails
    [0, 254, 254],  # 17: rail-raised
    [0, 68, 63]  # 18: rail-embedded
]


def decode_segmap(mask, colormap=RS19_COLORMAP):
    """
    Converts a 2D mask of class indices into a color image.

    Args:
        mask (np.ndarray): 2D array of class indices (HxW).
        colormap (list): RGB colors for each class.

    Returns:
        np.ndarray: HxWx3 RGB image.
    """
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, color in enumerate(colormap):
        color_mask[mask == idx] = color

    return color_mask

def draw_detection(image, box, label, on_track=False):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0) if on_track else (0, 0, 255)
    thickness = 2

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size, _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def overlay_segmentation_mask(image, mask, color=(0, 255, 255), alpha=0.4):
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask == 1] = color
    return cv2.addWeighted(mask_colored, alpha, overlay, 1 - alpha, 0)