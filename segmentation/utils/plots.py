# segmentation/utils/plots.py

import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_ious, val_ious, model_name, save_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title(f'Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss_{model_name}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.title(f'Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_dir}/acc_{model_name}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ious, label='Train IoU')
    plt.plot(epochs, val_ious, label='Val IoU')
    plt.title(f'IoU - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(f'{save_dir}/iou_{model_name}.png')
    plt.close()

    print(f"Saved all plots to {save_dir}/")