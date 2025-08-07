# fusion/models/loader.py

import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO  # YOLOv11 interface
from fusion.models.utils import load_checkpoint


def load_segmentation_model(config, device):
    """
    Load a DeepLabV3+ segmentation model.
    """
    model_type = config['segmentation_model_type'].lower()
    num_classes = config['segmentation_num_classes']
    checkpoint_path = config['segmentation_model_path']

    if model_type == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported segmentation model: {model_type}")

    model = load_checkpoint(model, checkpoint_path, device)
    return model.eval()


def load_detection_model(config):
    """
    Load a YOLOv11 model from a .pt file.
    """
    model_path = config['detection_model_path']
    yolo_model = YOLO(model_path)
    return yolo_model