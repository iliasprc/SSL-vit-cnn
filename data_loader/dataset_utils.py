import os

import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

from data_loader.cifar100 import cifar100


def select_dataset(config):
    test_params = {'batch_size' : config.dataloader.test.batch_size,
                   'shuffle'    : False,
                   'num_workers': 2}
    val_params = {'batch_size' : config.dataloader.val.batch_size,
                  'shuffle'    : config.dataloader.val.shuffle,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory' : True}

    train_params = {'batch_size' : config.dataloader.train.batch_size,
                    'shuffle'    : config.dataloader.train.shuffle,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory' : True}
    if config.dataset.name == 'CIFAR100':
        return cifar100(config)


import torchvision.transforms as transforms

imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size  # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                                   p=p_blur),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


def ssl_dataset(config):
    root_dir = os.path.join(config.cwd, config.dataset.input_data)

    train_transform = SimSiamTransform(32)
    # val_transform = SimSiamTransform(32)
    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.CIFAR100(root=root_dir, transform=train_transform, train=True,
                                                  download=True)

    feature_bank_dataset = torchvision.datasets.CIFAR100(root=root_dir, transform=val_transform, train=True,
                                                  download=True)
    valid_dataset = torchvision.datasets.CIFAR100(root=root_dir, transform=val_transform, train=True,
                                                  download=True)
    valid_size = 0.2
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    valid_sampler = SubsetRandomSampler(valid_idx)
    val_params = {'batch_size' : config.batch_size,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory' : False}

    train_params = {'batch_size' : config.batch_size,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory' : False}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, **train_params, sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, **val_params, sampler=valid_sampler
    )
    fbank_loader = torch.utils.data.DataLoader(
        valid_dataset, **val_params, sampler=train_sampler
    )
    return train_loader,fbank_loader, valid_loader,  train_dataset.classes
