# fusion/scripts/run_fusion.py

import os
import cv2
import torch
import argparse
from tqdm import tqdm

from fusion.models.loader import load_segmentation_model, load_detection_model
from fusion.engine import fuse_single_frame

def get_args():
    parser = argparse.ArgumentParser(description="Run segmentation + detection fusion.")
    parser.add_argument('--img_dir', type=str, required=True, help="Directory of input images.")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Directory to save fusion outputs.")
    parser.add_argument('--seg_weights', type=str, required=True, help="Path to segmentation model weights.")
    parser.add_argument('--yolo_weights', type=str, required=True, help="Path to YOLOv11 .pt weights.")
    parser.add_argument('--track_class_id', type=int, default=12, help="Class ID for 'rail-track'.")
    parser.add_argument('--export_video', action='store_true', help="Whether to export results as video.")
    parser.add_argument('--video_name', type=str, default='fused_output.mp4', help="Output video filename.")
    return parser.parse_args()


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("🔁 Loading models...")
    segmentor = load_segmentation_model(args.seg_weights, device=device)
    detector = load_detection_model(args.yolo_weights)

    # Image paths
    image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.png'))])

    # Video setup
    if args.export_video:
        first_img = cv2.imread(os.path.join(args.img_dir, image_files[0]))
        height, width, _ = first_img.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(args.output_dir, args.video_name)
        out_vid = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

    # Run fusion
    config = {"track_class_id": args.track_class_id}
    print(f"🚀 Processing {len(image_files)} frames...")
    for fname in tqdm(image_files):
        img_path = os.path.join(args.img_dir, fname)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = fuse_single_frame(image_rgb, segmentor, detector, config, device)

        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, result)

        if args.export_video:
            out_vid.write(result)

    if args.export_video:
        out_vid.release()
        print(f"🎥 Video saved to: {video_path}")

    print("✅ Fusion complete!")


if __name__ == "__main__":
    main()