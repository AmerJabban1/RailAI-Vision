# # detection/utils/dataset.py

import os
import torch
import torchvision
from PIL import Image
import numpy as np

class RailDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(image_dir, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(image_dir, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, "images", self.imgs[idx])
        label_path = os.path.join(self.image_dir, "labels", self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                img_w, img_h = img.size
                xmin = (x_center - width / 2) * img_w
                xmax = (x_center + width / 2) * img_w
                ymin = (y_center - height / 2) * img_h
                ymax = (y_center + height / 2) * img_h
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)