import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models.model import CVAE
import matplotlib.pyplot as plt


def visualize(model, labels, dev_str='cuda:0', figsize=(30, 30), save_path=None):
    images = model.generate(labels, dev_str).permute(0, 2, 3, 1).clip(0., 1.)    # (B, W, H, C)
    images = images.detach().cpu().numpy()
    plt.figure(figsize=figsize)
    num = len(labels)
    row = int(np.sqrt(num)) + 1
    for i in range(num):
        ax = plt.subplot(row, row, i+1)
        ax.set_title(f"Class: {labels[i].item()}")
        plt.imshow(images[i])
    if save_path is not None:
        plt.savefig(save_path)