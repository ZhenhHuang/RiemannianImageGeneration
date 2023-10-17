from models.basic import UNetEncoder, UNetDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import TimeEmbedding
from models.act_funcs import act_selector


class UNet(nn.Module):
    """
    Args:
        n_layers: the number of down-sample
    """

    def __init__(self, n_layers, in_channel, hidden_channels: list = None, out_channels=10,
                 act_func='relu', bilinear=False, use_attn=True):
        super(UNet, self).__init__()
        if hidden_channels is None:
            n_layers = 4
            hidden_channels = [64, 128, 256, 512, 1024]
        self.encoder = UNetEncoder(n_layers, in_channel, hidden_channels, act_func, bilinear)
        self.decoder = UNetDecoder(n_layers, hidden_channels[::-1], out_channels, act_func, bilinear, use_attn)

    def forward(self, x):
        x_list = self.encoder(x)
        out = self.decoder(x_list)
        return out


class TemporalUNet(nn.Module):
    """
        Args:
            n_layers: the number of down-sample
        """

    def __init__(self, n_layers, in_channel, n_classes, hidden_channels: list = None, out_channels=10,
                 cond_channel=16, time_channel=16, act_func='relu', bilinear=False, use_attn=True):
        super(TemporalUNet, self).__init__()
        if hidden_channels is None:
            n_layers = 4
            hidden_channels = [64, 128, 256, 512, 1024]
        in_channel = in_channel + time_channel + cond_channel
        self.encoder = UNetEncoder(n_layers, in_channel, hidden_channels, act_func, bilinear)
        self.decoder = UNetDecoder(n_layers, hidden_channels[::-1], out_channels, act_func, bilinear, use_attn)
        self.time_embedding = TimeEmbedding(time_channel, act_func=act_func)
        if cond_channel != 0:
            self.label_embedding = nn.Embedding(n_classes, cond_channel)

    def forward(self, t, x, y=None):
        """

        :param t: (T, ) -> (T, D_t) -> (T, 1, D_t, 1, 1)
        :param x: (T, B, C, H, W)
        :param y: None or (B, ) -> (B, D_y) -> (1, B, D_y, 1, 1)
        :return: (T, B, C, H, W)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if x.dim() == 4:
            x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        t = self.time_embedding(t).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, B, 1, H, W)
        if y is not None:
            y = self.label_embedding(y).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(T, 1, 1, H, W)
            x_list = self.encoder(torch.concat([t, x, y], dim=-3).reshape(T*B, -1, H, W))
        else:
            x_list = self.encoder(torch.concat([t, x], dim=-3).reshape(T * B, -1, H, W))
        out = self.decoder(x_list).reshape(T, B, -1, H, W)
        return out


class TemporalMLP(nn.Module):
    def __init__(self, n_layers, in_dim, n_classes, hidden_dims: list = None, out_dim=10,
                 cond_dim=16, time_channel=16, act_func='relu'):
        super(TemporalMLP, self).__init__()
        assert n_layers == len(hidden_dims) + 1, \
            f"hidden dimensions are not matching with the number of layers"
        self.layers = nn.ModuleList([nn.Flatten(-3, -1),
                                     nn.Linear(in_dim, hidden_dims[0])])
        for i in range(n_layers - 2):
            self.layers.append(act_selector(act_func))
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(act_selector(act_func))
        self.layers.append(nn.Linear(hidden_dims[-1] + cond_dim + time_channel, out_dim))
        self.time_embedding = TimeEmbedding(time_channel, act_func=act_func)
        if cond_dim != 0:
            self.label_embedding = nn.Embedding(n_classes, cond_dim)

    def forward(self, t, x, y=None):
        """

        :param t: (T, ) -> (T, D_t) -> (T, 1, D_t)
        :param x: (T, B, C, H, W)
        :param y: None or (B, ) -> (B, D_y) -> (1, B, D_y, 1, 1)
        :return: (T, B, C, H, W)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if x.dim() == 4:
            x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        for layer in self.layers[:-1]:
            x = layer(x)
        t = self.time_embedding(t).unsqueeze(1).repeat(1, B, 1)  # (T, B, D_t)
        if y is not None:
            y = self.label_embedding(y).unsqueeze(0).repeat(T, 1, 1)    # (T, B, D_y)
            x = self.layers[-1](torch.concat([t, x, y], dim=-1))
        else:
            x = self.layers[-1](torch.concat([t, x], dim=-1))
        out = x.reshape(T, B, -1, H, W)
        return out