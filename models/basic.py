import torch
import torch.nn as nn
import torch.nn.functional as F
from models.act_funcs import act_selector
from layers import UpSample, DownSample, DoubleConv3x3


class VAEEncoder(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dims: list, out_dim, cond_dim=0, act_func="relu"):
        super(VAEEncoder, self).__init__()
        assert n_layers == len(hidden_dims) + 1, \
            f"hidden dimensions are not matching with the number of layers"
        self.layers = nn.ModuleList([nn.Flatten(1, -1),
                                     nn.Linear(in_dim, hidden_dims[0])])
        for i in range(n_layers - 2):
            self.layers.append(act_selector(act_func))
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(act_selector(act_func))
        self.mean_enc = nn.Linear(hidden_dims[-1] + cond_dim, out_dim)
        self.log_std_enc = nn.Linear(hidden_dims[-1] + cond_dim, out_dim)

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_enc(x) if y is None else self.mean_enc(torch.cat([x, y], dim=-1))
        log_std = self.log_std_enc(x) if y is None else self.log_std_enc(torch.cat([x, y], dim=-1))
        return mean, log_std


class VAEDecoder(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dims: list, out_dim, cond_dim, CWH, act_func="relu"):
        super(VAEDecoder, self).__init__()
        assert n_layers == len(hidden_dims) + 1, \
            f"hidden dimensions are not matching with the number of layers"
        self.CWH = CWH
        self.layers = nn.ModuleList([nn.Linear(in_dim + cond_dim, hidden_dims[0])])
        for i in range(n_layers - 2):
            self.layers.append(act_selector(act_func))
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(act_selector(act_func))
        self.layers.append(nn.Linear(hidden_dims[-1], out_dim))

    def forward(self, x, y=None):
        x = self.layers[0](x) if y is None else self.layers[0](torch.cat([x, y], dim=-1))
        for layer in self.layers[1:]:
            x = layer(x)
        x = x.reshape(-1, *self.CWH)
        return x


class VAELoss(nn.Module):
    """
    Args:
        beta: >= 1.0
    """
    def __init__(self, beta=1.0):
        super(VAELoss, self).__init__()
        self.beta = beta

    def forward(self, x, x_rec, mean, log_std):
        l_rec = F.mse_loss(x, x_rec)
        l_kl = -0.5 * torch.sum(1 + 2 * log_std - mean ** 2 - (2 * log_std).exp(), dim=-1).mean()
        return l_rec + self.beta * l_kl


class UNetEncoder(nn.Module):
    """
    Args:
        n_layers: the number of down-sample
    """
    def __init__(self, n_layers, in_channel, hidden_channels: list = None, act_func='relu', bilinear=False):
        super(UNetEncoder, self).__init__()
        assert n_layers == len(hidden_channels) - 1
        factor = 2 if bilinear else 1
        self.layers = nn.ModuleList([DoubleConv3x3(in_channel, hidden_channels[0], act_func=act_func)])
        for i in range(n_layers):
            if i == (n_layers - 1):
                self.layers.append(DownSample(hidden_channels[i], hidden_channels[i + 1] // factor, act_func=act_func))
            else:
                self.layers.append(DownSample(hidden_channels[i], hidden_channels[i + 1], act_func=act_func))

    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return: List of down sample outputs in order
        """
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x)
        return out


class UNetDecoder(nn.Module):
    def __init__(self, n_layers, hidden_channels: list = None, out_channels=10, act_func='relu', bilinear=False):
        super(UNetDecoder, self).__init__()
        assert n_layers == len(hidden_channels) - 1
        factor = 2 if bilinear else 1
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == (n_layers - 1):
                self.layers.append(UpSample(hidden_channels[i], hidden_channels[i + 1],
                                            act_func=act_func, bilinear=bilinear))
            else:
                self.layers.append(UpSample(hidden_channels[i], hidden_channels[i + 1] // factor,
                                            act_func=act_func, bilinear=bilinear))
        self.layers.append(nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=1))

    def forward(self, x_list):
        x = x_list[-1]
        for i, layer in enumerate(self.layers[:-1]):
            x_prev = x_list[-(i+2)]
            x = layer(x, x_prev)
        x = self.layers[-1](x)
        return x








