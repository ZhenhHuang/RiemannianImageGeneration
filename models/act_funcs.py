import torch
import torch.nn as nn
import torch.nn.functional as F


def act_selector(act_func):
    if act_func.lower() == 'relu':
        return nn.ReLU()
    elif act_func.lower() == 'elu':
        return nn.ELU()
    elif act_func.lower() == 'swish':
        return nn.SiLU()
    elif act_func.lower() == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    else:
        raise NotImplementedError(f'The activation {act_func} is not implemented')