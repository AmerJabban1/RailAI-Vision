# segmentation/scripts/train.py

import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from segmentation.datasets.railsem19 import RailSem19Dataset
from segmentation.models.factory import get_model
from segmentation.utils.metrics import calculate_accuracy, calculate_iou
from segmentation.utils.plots import plot_metrics
from segmentation.utils.common import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model on RailSem19")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataloaders(image_dir, mask_dir, batch_size):
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.4, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    train_dataset = RailSem19Dataset(image_dir, mask_dir, train_imgs, image_transforms, mask_transforms)
    val_dataset = RailSem19Dataset(image_dir, mask_dir, val_imgs, image_transforms, mask_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def train(model, train_loader, val_loader, config, model_name, device):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_iou = 0
    results_dir = config['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_ious, val_ious = [], []

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        model.train()
        total_loss, total_acc, total_iou = 0, 0, 0

        for images, masks in tqdm(train_loader, desc='Training'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, masks)
            total_iou += calculate_iou(outputs, masks)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        avg_train_iou = total_iou / len(train_loader)

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        train_ious.append(avg_train_iou)

        # Validation
        model.eval()
        total_loss, total_acc, total_iou = 0, 0, 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                total_loss += loss.item()
                total_acc += calculate_accuracy(outputs, masks)
                total_iou += calculate_iou(outputs, masks)

        avg_val_loss = total_loss / len(val_loader)
        avg_val_acc = total_acc / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)

        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        val_ious.append(avg_val_iou)

        scheduler.step(avg_val_loss)

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), os.path.join(results_dir, f'best_{model_name}.pth'))
            print(f"Saved best model with IoU: {best_iou:.4f}")

    plot_metrics(train_losses, val_losses, train_accs, val_accs, train_ious, val_ious, model_name, save_dir=results_dir)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get('seed', 42))
    device = get_device()

    model = get_model(
        config['model_name'],
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('num_classes', 19),
        pretrained=config.get('pretrained', True)
    )
    model = nn.DataParallel(model).to(device)

    train_loader, val_loader = prepare_dataloaders(config['image_dir'], config['mask_dir'], config['batch_size'])
    train(model, train_loader, val_loader, config, config['model_name'], device)