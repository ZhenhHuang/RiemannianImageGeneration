import torch
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, dims):
        super(Reshape, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.reshape(*self.dims)