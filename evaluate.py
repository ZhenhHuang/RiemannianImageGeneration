import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models.model import CVAE
import matplotlib.pyplot as plt


def visualize(model, labels, dev_str='cuda:0', figsize=(30, 30), save_path=None):
    images = model.generate(labels, dev_str).permute(0, 2, 3, 1) * 0.5 + 0.5   # (B, W, H, C)
    images = images.detach().cpu().numpy()
    if images.shape[-1] == 1:
        cmap = 'gray'
    else:
        cmap = None
    plt.figure(figsize=figsize)
    num = len(labels)
    row = int(np.sqrt(num)) + 1
    for i in range(num):
        ax = plt.subplot(row, row, i+1)
        ax.set_title(f"Class: {labels[i].item()}")
        plt.imshow(images[i].clip(0, 1), cmap=cmap)
    if save_path is not None:
        plt.savefig(save_path)


def visualize_process(model, labels, dev_str='cuda:0', figsize=(30, 30), ode_steps=10, save_path=None):
    """Only for Flow-based model and Diffusion-based model"""
    images, vec_field = model.generate(labels, dev_str, ode_steps, return_steps=True).permute(0, 1, 3, 4, 2)  # (T, B, W, H, C)
    images = images.detach().cpu().numpy()
    plt.figure(figsize=figsize)
    row = len(labels)
    col = ode_steps
    for i in range(row):
        for j in range(col):
            ax = plt.subplot(row, col, (i + 1) * (j + 1))
            ax.set_title(f"Class: {labels[i].item()}")
            plt.imshow(images[j, i])
    if save_path is not None:
        plt.savefig(save_path)


def interpolate(model, x_1, x_2, y_1=None, y_2=None, steps=10, dev_str='cuda:0', figsize=(30, 30), save_path=None):
    z_1, y_emb_1 = model.encode(x_1, y_1)[:2]  # (B, D)
    z_2, y_emb_2 = model.encode(x_2, y_2)[:2]
    z_t = torch.stack([z_1 + (z_2 - z_1) * t for t in torch.linspace(0, 1, steps)])  # (T, B, D)
    y_t = torch.stack([y_emb_1 + (y_emb_2 - y_emb_1) * t for t in torch.linspace(0, 1, steps)])  # (T, B, D)
    T, B, D = z_t.shape
    x_t = model.decode(z_t.reshape(-1, D), y_t.reshape(-1, D))
    x_t = x_t.reshape(T, B, *x_t.shape[1:])
    plt.figure(figsize=figsize)
    for i in range(B):
        for j in range(T):
            ax = plt.subplot(B, T, (i + 1) * (j + 1))
            plt.imshow(x_t[j, i])
    if save_path is not None:
        plt.savefig(save_path)