# scripts/run_depth_fusion_midas.py

"""
Offline detection + segmentation + depth estimation using MiDaS model.
"""

import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from depth.models.loader import load_yolo, load_deeplab, load_midas
from depth.models.utils import get_depth_at_bbox, calibrate_depth
from depth.engine import fuse_outputs

# ----------------------------
# Configuration
# ----------------------------
INPUT_DIR = "images"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.4
TRACK_CLASS_ID = 12
RAIL_GAUGE_METERS = 1.435  # For calibration

CLASS_NAMES = ["Person", "Traffic Sign/Signal", "Vehicle", "Obstacle", "Rail"]

# ----------------------------
# Preprocessing
# ----------------------------
# MiDaS transform
midas_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------------------
# Main
# ----------------------------
def process_image(image_path, yolo_model, seg_model, depth_model):
    filename = os.path.basename(image_path)
    print(f"[INFO] Processing: {filename}")

    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[ERROR] Could not read: {image_path}")
        return

    h, w = bgr.shape[:2]

    # MiDaS depth inference
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_input = midas_transform(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw_depth = depth_model(depth_input)
        depth_resized = torch.nn.functional.interpolate(
            raw_depth.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Fuse results (will handle seg + det + overlay)
    fused = fuse_outputs(
        image=bgr,
        depth_map=depth_resized,
        yolo_model=yolo_model,
        seg_model=seg_model,
        device=DEVICE,
        class_names=CLASS_NAMES,
        track_class_id=TRACK_CLASS_ID,
        conf_threshold=CONF_THRESHOLD,
        calibrate_depth=True,
        rail_gauge=RAIL_GAUGE_METERS
    )

    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, fused)
    print(f"[INFO] Saved: {out_path}")

# ----------------------------
# Entry Point
# ----------------------------
def main():
    yolo_model = load_yolo().to(DEVICE)
    seg_model = load_deeplab().to(DEVICE).eval()
    depth_model = load_midas().to(DEVICE).eval()

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        print("[ERROR] No input images found.")
        return

    for file in image_files:
        process_image(os.path.join(INPUT_DIR, file), yolo_model, seg_model, depth_model)


if __name__ == "__main__":
    main()