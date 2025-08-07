# segmentation/utils/metrics.py

import torch
import numpy as np

def calculate_accuracy(pred, target, ignore_index=255):
    pred = torch.argmax(pred, dim=1)
    mask = target != ignore_index
    correct = (pred == target) & mask
    return correct.sum().item() / mask.sum().item()

def calculate_iou(pred, target, num_classes=19, ignore_index=255):
    pred = torch.argmax(pred, dim=1)
    iou_list = []
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(intersection / union)
    valid_iou = [iou for iou in iou_list if not np.isnan(iou)]
    return np.mean(valid_iou) if valid_iou else 0.0