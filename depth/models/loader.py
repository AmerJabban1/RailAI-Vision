# depth/models/loader.py

import torch
from ultralytics import YOLO
from depth.utils.deep_lab_model import get_deeplab_model


def load_yolo(model_path, device):
    model = YOLO(model_path)
    model.fuse()
    return model


def load_deeplab(model_path, num_classes, device):
    model = get_deeplab_model(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)

    if list(state_dict.keys())[0].startswith("module."):
        from collections import OrderedDict
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_midas(device):
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True).to(device).eval()
    return model