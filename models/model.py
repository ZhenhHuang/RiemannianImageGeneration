import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import VAEEncoder, VAEDecoder, VAELoss, UNetEncoder, UNetDecoder


def sample(mu, log_std):
    eps = torch.randn_like(mu).to(mu.device)
    return mu + eps * log_std.exp()


class CVAE(nn.Module):
    """
    Args:
    in_dim: the dimension of input data, i.e., C*H*W or C
    out_dim: the dimension of latent variable
    """

    def __init__(self, n_layers, n_classes, in_dim, hidden_dims: list, out_dim,
                 cond_dim, CWH: tuple, act_func="relu"):
        super(CVAE, self).__init__()
        assert in_dim == CWH[0] or in_dim == CWH[0] * CWH[1] * CWH[2]
        self.latent_dim = out_dim
        self.encoder = VAEEncoder(n_layers, in_dim, hidden_dims, out_dim, cond_dim, act_func)
        self.decoder = VAEDecoder(n_layers, out_dim, hidden_dims[::-1], in_dim, cond_dim, CWH, act_func)
        self.label_enc = nn.Embedding(n_classes, cond_dim)

    def encode(self, x, y=None):
        y = self.label_enc(y)
        mean, log_std = self.encoder(x, y)
        z = sample(mean, log_std)
        return z, y, mean, log_std

    def decode(self, x, y=None):
        return self.decoder(x, y)

    def forward(self, x, y=None):
        """
        :param x: (B, C, H, W) or (B, H, W)
        :param y: (B, )
        :return:
        """
        z, y, mean, log_std = self.encode(x, y)
        z = self.decode(z, y)
        return z, mean, log_std

    @torch.no_grad()
    def generate(self, labels, dev_str='cuda:0'):
        """
        :param dev_str: device string, 'cpu' or 'cuda:{id}'
        :param labels: Tensor(B, )
        :return: z: (B, C, H, W)
        """
        device = torch.device(dev_str)
        labels = labels.to(device)
        y = self.label_enc(labels)
        z = torch.randn(labels.shape[0], self.latent_dim).to(device)
        z = self.decoder(z, y)
        return z

    def loss(self, x, x_rec, mean, log_std, beta=1.0):
        return VAELoss(beta)(x, x_rec, mean, log_std)


class UNet(nn.Module):
    """
    Args:
        n_layers: the number of down-sample
    """
    def __init__(self, n_layers, in_channel, hidden_channels: list = None, out_channels=10, act_func='relu', bilinear=False):
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


class 