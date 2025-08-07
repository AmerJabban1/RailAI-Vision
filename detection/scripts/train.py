# detection/scripts/train.py

import argparse
import yaml
import os
from ultralytics import YOLO
import torch

def train_yolo(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = YOLO(config['model_arch'])

    model.train(
        data=config['data_yaml'],
        epochs=config.get('epochs', 50),
        imgsz=config.get('imgsz', 640),
        batch=config.get('batch', 16),
        device=config.get('device', 0),
        project=config.get('project', './runs/train'),
        name=config.get('name', 'exp'),
        exist_ok=True,
        verbose=True,
        plots=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a YOLO model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    train_yolo(args.config)