# segmentation/datasets/railsem19.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class RailSem19Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = torch.from_numpy(np.array(mask)).long()
        return image, mask