from models.basic import UNetEncoder, UNetDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import TimeEmbedding


class UNet(nn.Module):
    """
    Args:
        n_layers: the number of down-sample
    """

    def __init__(self, n_layers, in_channel, hidden_channels: list = None, out_channels=10,
                 act_func='relu', bilinear=False):
        super(UNet, self).__init__()
        if hidden_channels is None:
            n_layers = 4
            hidden_channels = [64, 128, 256, 512, 1024]
        self.encoder = UNetEncoder(n_layers, in_channel, hidden_channels, act_func, bilinear)
        self.decoder = UNetDecoder(n_layers, hidden_channels[::-1], out_channels, act_func, bilinear)

    def forward(self, x):
        x_list = self.encoder(x)
        out = self.decoder(x_list)
        return out


class TemporalUNet(nn.Module):
    """
        Args:
            n_layers: the number of down-sample
        """

    def __init__(self, n_layers, in_channel, hidden_channels: list = None, out_channels=10,
                 time_steps=10, time_channel=16, act_func='relu', bilinear=False):
        super(TemporalUNet, self).__init__()
        if hidden_channels is None:
            n_layers = 4
            hidden_channels = [64, 128, 256, 512, 1024]
        in_channel = in_channel + time_channel
        self.encoder = UNetEncoder(n_layers, in_channel, hidden_channels, act_func, bilinear)
        self.decoder = UNetDecoder(n_layers, hidden_channels[::-1], out_channels, act_func, bilinear)
        self.time_embedding = TimeEmbedding(time_channel, act_func=act_func)

    def forward(self, t, x, y=None):
        """

        :param t: (T, ) -> (T, D_t) -> (T, 1, D_t, 1, 1)
        :param x: (T, B, C, H, W)
        :param y: None
        :return: (T, B, C, H, W)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if x.dim() == 4:
            x = x.unsqueeze(0)
        T, B, C, H, W = x.shape
        t = self.time_embedding(t).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, B, 1, H, W)
        x_list = self.encoder(torch.concat([x, t], dim=-3).reshape(T*B, -1, H, W))
        out = self.decoder(x_list).reshape(T, B, -1, H, W)
        return out