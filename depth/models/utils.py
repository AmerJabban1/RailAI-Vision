# depth/models/utils.py

import numpy as np
import cv2

def get_depth_at_bbox(depth_map, x1, y1, x2, y2):
    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return -1
    valid = crop[np.isfinite(crop) & (crop > 0)]
    return float(np.median(valid)) if valid.size > 0 else -1


def calibrate_depth(depth_map, rail_mask, known_distance=1.435):
    if rail_mask.sum() == 0:
        return depth_map
    rail_depths = depth_map[rail_mask > 0]
    if rail_depths.size > 0:
        median_rail = np.median(rail_depths[rail_depths > 0])
        if median_rail > 0:
            return depth_map * (known_distance / median_rail)
    return depth_map