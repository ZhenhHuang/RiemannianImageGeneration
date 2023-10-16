import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import VAELoss, VAEEncoder, VAEDecoder
from ode_solvers import integrator, solve_geodesic


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
                 cond_dim, CHW: tuple, act_func="relu"):
        super(CVAE, self).__init__()
        assert in_dim == CHW[0] or in_dim == CHW[0] * CHW[1] * CHW[2]
        self.latent_dim = out_dim
        self.encoder = VAEEncoder(n_layers, in_dim, hidden_dims, out_dim, cond_dim, act_func)
        self.decoder = VAEDecoder(n_layers, out_dim, hidden_dims[::-1], in_dim, cond_dim, CHW, act_func)
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


class FlowMatching(nn.Module):
    """
    Args:
        vector_field(t, x, y: Optional = None): a network with input shape (T, ) + (T, B, C, H ,W)
        and output shape (T, B, C, H, W)
    """
    def __init__(self, vector_field, CHW: tuple, kappa=0.):
        super(FlowMatching, self).__init__()
        self.kappa = torch.tensor(kappa)
        self.vector_field = vector_field
        self.size = CHW

    def forward(self, x, y=None, ode_steps: int = 200):
        """

        :param x: (B, C, H, W)
        :param y: (B, )
        :param ode_steps: int
        :return: learned vector field, real vector field with shape (T, B, C, H, W)
        """
        t = torch.linspace(0, 1, steps=ode_steps).to(x.device)
        x1 = x
        x0 = torch.randn_like(x1).to(x.device)
        xt, dxt_dt = solve_geodesic(t, x0, x1, kappa=self.kappa)
        vt = self.vector_field(t, xt, y)
        return vt, dxt_dt

    def loss(self, x, vt, dxt_dt):
        return F.mse_loss(vt, dxt_dt)

    @torch.no_grad()
    def generate(self, labels, dev_str='cuda:0', ode_steps: int = 200, return_steps=False):
        """
        :param return_steps: whether return path tracker
        :param ode_steps: ode solver steps
        :param dev_str: device string, 'cpu' or 'cuda:{id}'
        :param labels: Tensor(B, )
        :return: z: (B, C, H, W)
        """
        device = torch.device(dev_str)
        t = torch.linspace(0, 1, steps=ode_steps).to(device)
        labels = labels.to(device)
        z = torch.randn(labels.shape[0], *self.size).to(device)
        xt, dxt_dt = integrator(lambda tt, xx: self.vector_field(tt, xx, labels), z, t, self.kappa)
        if return_steps:
            return xt, dxt_dt
        else:
            return xt[-1]


# if __name__ == '__main__':
#     vf = TemporalUNet(4, 3, out_channels=3)
#     model = FlowMatching(vf)
#     x = torch.randn(2, 3, 32, 32)
#     vt, dxt_dt = model(x)