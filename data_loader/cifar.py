import os

import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

size = 32
train_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomResizedCrop((size), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def cifar100(config):
    root_dir = os.path.join(config.cwd, config.dataset.input_data)
    train_dataset = torchvision.datasets.CIFAR100(root=root_dir, transform=train_transform, train=True,
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
    test_params = {'batch_size' : config.batch_size,
                   'shuffle'    : False,
                   'num_workers': 2}
    val_params = {'batch_size' : config.batch_size,
                  'shuffle'    : False,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory' : True}

    train_params = {'batch_size' : config.batch_size,
                    'shuffle'    : config.dataloader.train.shuffle,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory' : True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, **train_params, sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, **val_params, sampler=valid_sampler
    )
    test_dataset = torchvision.datasets.CIFAR100(root=root_dir, transform=val_transform, train=False,
                                                 download=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, **val_params
    )
    return train_loader, valid_loader, test_loader, train_dataset.classes


def cifar10(config):
    root_dir = os.path.join(config.cwd, config.dataset.input_data)
    train_dataset = torchvision.datasets.CIFAR10(root=root_dir, transform=train_transform, train=True,
                                                 download=True)
    valid_dataset = torchvision.datasets.CIFAR10(root=root_dir, transform=val_transform, train=True,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=root_dir, transform=val_transform, train=False,
                                                download=True)

    valid_size = 0.2
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_params = {'batch_size' : config.batch_size,
                   'shuffle'    : False,
                   'num_workers': 2}
    val_params = {'batch_size' : config.batch_size,
                  'shuffle'    : False,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory' : True}

    train_params = {'batch_size' : config.batch_size,
                    'shuffle'    : config.dataloader.train.shuffle,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory' : True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, **train_params, sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, **val_params, sampler=valid_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, **val_params
    )
    return train_loader, valid_loader, test_loader, train_dataset.classes



def stl10(config):
    size =224
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomResizedCrop((size), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_params = {'batch_size' : config.batch_size,
                   'shuffle'    : False,
                   'num_workers': 2}
    val_params = {'batch_size' : config.batch_size,
                  'shuffle'    : False,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory' : True}

    train_params = {'batch_size' : config.batch_size,
                    'shuffle'    : config.dataloader.train.shuffle,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory' : True}
    #SIZE = int(config.shape)
    root_dir = os.path.join(config.cwd, config.dataset.input_data)
    unlabelled_dataset = torchvision.datasets.STL10(root=root_dir, transform=train_transform, split='train+unlabeled',
                                               download=True)

    train_dataset = torchvision.datasets.STL10(root=root_dir, transform=train_transform, split='train',
                                             download=True)
    test_dataset = torchvision.datasets.STL10(root=root_dir, transform=val_transform, split='test',
                                              download=True)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, **train_params
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, **val_params
    )
    return train_loader, test_loader,None, train_dataset.classes