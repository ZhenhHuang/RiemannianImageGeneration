import torch
import torch.nn as nn
import torch.nn.functional as F
from models.act_funcs import act_selector


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
