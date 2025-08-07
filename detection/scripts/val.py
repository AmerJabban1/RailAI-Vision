# detection/scripts/val.py

import argparse
from ultralytics import YOLO

def validate(model_path, data_yaml, imgsz=640):
    model = YOLO(model_path)
    model.val(data=data_yaml, imgsz=imgsz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a YOLO model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained weights')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    args = parser.parse_args()
    validate(args.model, args.data, args.imgsz)