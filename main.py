import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='ImageGeneration')

# Experiment settings
parser.add_argument('--dataset', type=str, default='USPS')
parser.add_argument('--root_path', type=str, default='C:/Users/98311/Desktop/dataset/Images')
parser.add_argument('--training', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--verbose_freq', type=int, default=5)
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--exp_iters', type=int, default=1)
parser.add_argument('--version', type=str, default="run")
parser.add_argument('--log_path', type=str, default="./results/v1.log")
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--download', type=bool, default=False)
parser.add_argument('--results_path', type=str, default="results/result.png")

parser.add_argument('--model_type', type=str, default='flow_based', choices=['vae_based', 'flow_based'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--save_id', type=str, default='cave.pt')
parser.add_argument('--pre_epochs', type=int, default=10)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--act_func', type=str, default='relu')
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[300, 500], help='dimension of hidden layers')
parser.add_argument('--out_dim', type=int, default=10, help='dimension of latent space')
parser.add_argument('--cond_dim', type=int, default=16, help='dimension of embedding labels')
parser.add_argument('--time_dim', type=int, default=16, help='dimension of time embeddings')
parser.add_argument('--ode_steps', type=int, default=50, help='steps for solving ode')
parser.add_argument('--bilinear', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--w_decay', type=float, default=1e-4)
parser.add_argument('--kappa', type=float, default=-1.0, help='curvature of simple manifolds')
parser.add_argument('--beta', type=float, default=1.0, help='coefficient of Beta-VAE')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--learning_rate', type=float, default=1e-4)

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multile gpus')

configs = parser.parse_args()
log_path = f"./results/{configs.version}/{configs.dataset}.log"
configs.log_path = log_path
results_path = f"./results/{configs.version}/{configs.dataset}.pdf"
configs.results_path = results_path
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.version}"):
    os.mkdir(f"./results/{configs.version}")
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
if configs.training:
    exp.train()
test_labels = torch.randint(0, 10, (16,))
logger.info(f"Generating class of {test_labels}")
exp.eval(test_labels)
torch.cuda.empty_cache()