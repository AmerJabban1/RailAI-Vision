# fusion/models/utils.py

import torch
import os


def load_checkpoint(model, checkpoint_path, device):
    """
    Loads model weights into a PyTorch model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    return model