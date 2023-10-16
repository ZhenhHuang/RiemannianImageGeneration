import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.datasets import MNIST, USPS, ImageNet
import numpy as np
import os
from PIL import Image


def load_handwrite_digit(configs, train=True, download=False, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5)])
    if configs.dataset == 'USPS':
        dataset = USPS(f"{configs.root_path}/USPS", train=train, transform=transform, download=download)
    elif configs.dataset == 'MNIST':
        dataset = MNIST(configs.root_path, train=train, transform=transform, download=download)
    else:
        raise NotImplementedError
    return dataset


def load_CIFAR_10(configs, train=True, download=False, transform=None):
    root_path = f"{configs.root_path}/CIFAR-10"
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    CIFAR = torchvision.datasets.CIFAR10
    if transform is None and train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((-90, 90), interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
    elif transform is None and not train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
    cifar_data_test = CIFAR(root_path, train=train, download=download, transform=transform)
    return cifar_data_test


def load_CIFAR_100(configs, train=True, download=False, transform=None):
    root_path = f"{configs.root_path}/CIFAR-10"
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    CIFAR = torchvision.datasets.CIFAR100
    if transform is None and train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((-90, 90), interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
    elif transform is None and not train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
    cifar_data = CIFAR(root_path, train=train, download=download, transform=transform)
    return cifar_data


def load_ImageNet(configs, train=True, download=False, transform=None):
    root_path = f"{configs.root_path}/"
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    if transform is None and train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((-90, 90), interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
    elif transform is None and not train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
    if train:
        split = 'train'
    else:
        split = 'val'
    cifar_data = ImageNet(root_path, split=split, download=download, transform=transform)
    return cifar_data




def getLoader(configs, flag, transform=None):
    if flag == 'train':
        batch_size = configs.batch_size
        shuffle = True
        drop_last = True
        train = True
    elif flag == 'test' or flag == 'pred':
        batch_size = 1
        shuffle = False
        drop_last = False
        train = False
    else:
        batch_size = configs.batch_size
        shuffle = False
        drop_last = True
        train = False

    dataset = getDataset(configs, train, transform)
    print(f"{flag}: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataset, data_loader


def getDataset(configs, train, transform):
    if configs.dataset in ['MNIST', 'USPS']:
        return load_handwrite_digit(configs, train=train, download=configs.download)

    elif configs.dataset == 'cifar-10':
        return load_CIFAR_10(configs, train=train, download=configs.download)

    elif configs.dataset == 'cifar-100':
        return load_CIFAR_100(configs, train=train, download=configs.download)

    else:
        raise NotImplementedError