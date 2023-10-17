import torch
import torch.nn.functional as F
import torch.nn as nn
from models.model import CVAE, FlowMatching
from models.backbone import TemporalUNet, TemporalMLP
from data_handler.dataset import datasize
import inspect


def add_configs_params(func, args: dict, params: dict = None):
    sig = inspect.signature(func)
    for key in list(sig.parameters.keys())[len(params):]:
        params[key] = args[key]
    return params


def add_func_params_ordered(func, tuple_params, params: dict = None):
    sig = inspect.signature(func)
    for i, key in enumerate(list(sig.parameters.keys())[len(params):]):
        if i == len(tuple_params):
            break
        params[key] = tuple_params[i]
    return params


def vector_field(configs, img_args):
    size = img_args['size']
    if configs.vector_field == 'unet':
        return TemporalUNet(n_layers=configs.n_layers, in_channel=size[0], n_classes=img_args['n_classes'],
                                        hidden_channels=configs.hidden_dims, out_channels=size[0],
                                        cond_channel=configs.cond_dim, time_channel=configs.time_dim,
                                          act_func=configs.act_func, bilinear=configs.bilinear,
                                          use_attn=configs.use_attn)
    elif configs.vector_field == 'mlp':
        return TemporalMLP(n_layers=configs.n_layers, in_dim=img_args["in_dim"], n_classes=img_args['n_classes'],
                                        hidden_dims=configs.hidden_dims, out_dim=img_args["in_dim"],
                                        cond_dim=configs.cond_dim, time_channel=configs.time_dim,
                                        act_func=configs.act_func)


class Generator(nn.Module):
    def __init__(self, configs):
        super(Generator, self).__init__()
        self.configs = configs
        args = datasize[self.configs.dataset]
        if self.configs.model_type == 'vae_based':
            self.model = CVAE(n_layers=self.configs.n_layers, n_classes=args['n_classes'], in_dim=args["in_dim"],
                              hidden_dims=self.configs.hidden_dims, out_dim=self.configs.out_dim,
                              cond_dim=self.configs.cond_dim, CHW=args['size'], act_func=self.configs.act_func)
        elif self.configs.model_type == 'flow_based':
            self.model = FlowMatching(
                vector_field=vector_field(configs, args),
                CHW=args['size'],
                kappa=self.configs.kappa
            )
        else:
            raise NotImplementedError

    def forward(self, x, y=None):
        params = {'x': x, 'y': y}
        params = add_configs_params(self.model.forward, vars(self.configs), params)
        return self.model(**params)

    def loss(self, x, y=None):
        return_tuple = self.forward(x, y)
        params = {'x': x}
        params = add_func_params_ordered(self.model.loss, return_tuple, params)
        params = add_configs_params(self.model.loss, vars(self.configs), params)
        return self.model.loss(**params)

    @torch.no_grad()
    def generate(self, labels, dev_str='cuda:0'):
        params = {'labels': labels, 'dev_str': dev_str}
        params = add_configs_params(self.model.generate, vars(self.configs), params)
        return self.model.generate(**params)