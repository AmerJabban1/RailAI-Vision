# segmentation/utils/common.py

import os
import torch
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')