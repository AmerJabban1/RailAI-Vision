# segmentation/scripts/evaluate.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import yaml
import numpy as np
from tqdm import tqdm

from segmentation.datasets.railsem19 import RailSem19Dataset
from segmentation.models.factory import get_model
from segmentation.utils.metrics import calculate_accuracy, calculate_iou
from segmentation.utils.common import set_seed, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model on the RailSem19 test set")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth file)")
    parser.add_argument("--data_root", required=True, help="Root directory for the dataset")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = get_device()

    # Load test data
    image_dir = os.path.join(args.data_root, "jpgs", cfg["split"])
    mask_dir = os.path.join(args.data_root, "uint8", cfg["split"])
    test_list = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    image_transforms = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])
    mask_transforms = transforms.Resize(cfg["image_size"], interpolation=transforms.InterpolationMode.NEAREST)

    test_dataset = RailSem19Dataset(image_dir, mask_dir, test_list, image_transforms, mask_transforms)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = get_model(cfg["model"], in_channels=3, out_channels=cfg["num_classes"], pretrained=False)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    total_acc = 0.0
    total_iou = 0.0
    steps = 0

    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating") as t:
            for images, masks in t:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                total_acc += calculate_accuracy(outputs, masks)
                total_iou += calculate_iou(outputs, masks, num_classes=cfg["num_classes"])
                steps += 1
                t.set_postfix({"Accuracy": total_acc/steps, "IoU": total_iou/steps})

    print("\nFinal Test Results:")
    print(f"  Accuracy: {total_acc / steps:.4f}")
    print(f"  Mean IoU: {total_iou / steps:.4f}")

if __name__ == "__main__":
    main()