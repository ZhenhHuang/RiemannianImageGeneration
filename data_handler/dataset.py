import torch
from torchvision.transforms.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader


datasize = {
    'MNIST': {"size": (1, 28, 28), "in_dim": 28 * 28},
    'USPS': {"size": (1, 16, 16), "in_dim": 16 * 16},
    'cifar-10': {"size": (3, 32, 32), "in_dim": 3 * 32 * 32},
    'cifar-100': {"size": (3, 32, 32), "in_dim": 3 * 32 * 32},
}