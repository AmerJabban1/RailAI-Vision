# detection/scripts/predict.py

import argparse
from ultralytics import YOLO

def predict(model_path, source, conf=0.25, imgsz=640):
    model = YOLO(model_path)
    model.predict(source=source, conf=conf, imgsz=imgsz, save=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference using a trained YOLO model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained weights')
    parser.add_argument('--source', type=str, required=True, help='Folder of images to predict on')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    args = parser.parse_args()
    predict(args.model, args.source, args.conf, args.imgsz)