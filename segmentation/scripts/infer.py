# segmentation/scripts/infer.py

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from segmentation.models.factory import get_model
from segmentation.utils.common import set_seed, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a segmentation model")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--image", required=True, help="Path to a single input image")
    parser.add_argument("--output", default="prediction.png", help="Path to save the output segmentation map")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = get_device()

    model = get_model(cfg["model"], in_channels=3, out_channels=cfg["num_classes"], pretrained=False)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    image = Image.open(args.image).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Save prediction as image
    plt.figure(figsize=(8, 6))
    plt.imshow(pred, cmap='jet')
    plt.axis('off')
    plt.title("Predicted Segmentation")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved prediction to {args.output}")

if __name__ == "__main__":
    main()