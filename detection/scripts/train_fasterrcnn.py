# # detection/scripts/train_fasterrcnn.py

import os
import argparse
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
import torchvision
from detection.utils.engine import train_one_epoch, evaluate
from detection.utils.dataset import RailDataset
from detection.utils.utils import collate_fn, save_model, get_num_classes


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    return model


def main(args):
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = RailDataset(cfg["train_dir"], transforms=T.ToTensor())
    val_ds = RailDataset(cfg["val_dir"], transforms=T.ToTensor())

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = get_model(cfg["num_classes"]).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(cfg["epochs"]):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, val_loader, device=device)

        save_model(model, os.path.join(cfg["save_dir"], f"fasterrcnn_epoch{epoch+1}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args)