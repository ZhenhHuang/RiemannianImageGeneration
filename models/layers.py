import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, dims):
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class Reshape(nn.Module):
    def __init__(self, dims):
        super(Reshape, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.reshape(*self.dims)