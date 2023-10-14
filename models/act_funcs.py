import torch
import torch.nn as nn
import torch.nn.functional as F


def act_selector(act_func):
    if act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'silu':
        return nn.SiLU()
    else:
        raise NotImplementedError