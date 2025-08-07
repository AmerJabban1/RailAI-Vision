# depth/engine.py

import cv2
import numpy as np
import torch
import os
from depth.models.utils import get_depth_at_bbox, calibrate_depth
from depth.utils.visualization import draw_detection, overlay_segmentation_mask


def fuse_outputs(image_bgr, seg_mask, depth_map, detections, class_names, track_class_id=12, conf_threshold=0.4):
    rail_mask = (seg_mask == track_class_id).astype(np.uint8)
    height, width = image_bgr.shape[:2]
    rail_mask = cv2.resize(rail_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    boxes = detections.boxes.xyxy.cpu().numpy() if detections.boxes else []
    scores = detections.boxes.conf.cpu().numpy() if detections.boxes else []
    labels = detections.boxes.cls.cpu().numpy().astype(int) if detections.boxes else []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        confidence = scores[i]
        class_id = labels[i]

        if confidence < conf_threshold:
            continue

        object_mask = np.zeros(rail_mask.shape, dtype=np.uint8)
        object_mask[y1:y2, x1:x2] = 1
        overlap = cv2.bitwise_and(object_mask, rail_mask)
        on_track = np.any(overlap > 0)

        depth_meters = get_depth_at_bbox(depth_map, x1, y1, x2, y2)
        depth_str = f"{depth_meters:.1f}m" if depth_meters > 0 else "N/A"

        label = f"{class_names[class_id] if class_id < len(class_names) else 'Class'}: {confidence*100:.1f}%, {depth_str}"
        draw_detection(image_bgr, box, label, on_track=on_track)

    return overlay_segmentation_mask(image_bgr, rail_mask, color=(0, 255, 255), alpha=0.4)