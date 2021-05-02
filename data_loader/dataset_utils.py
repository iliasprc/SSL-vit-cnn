import os

import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler


from data_loader.random_dataset import RandomDataset




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
    SIZE = int(config.shape)
    root_dir = os.path.join(config.cwd, config.dataset.input_data)
    if config.dataset.name == 'CIFAR100':

        train_transform = SimSiamTransform(SIZE)
        # val_transform = SimSiamTransform(32)
        val_transform = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=root_dir, transform=train_transform, train=True,
                                                      download=True)

        feature_bank_dataset = torchvision.datasets.CIFAR100(root=root_dir, transform=train_transform, train=True,
                                                             download=True)
        valid_dataset = torchvision.datasets.CIFAR100(root=root_dir, transform=val_transform, train=False,
                                                      download=True)
        valid_size = 0.2
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        np.random.shuffle(indices)
        #
        # train_idx, valid_idx = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_idx)
        #
        # valid_sampler = SubsetRandomSampler(valid_idx)
        val_params = {'batch_size' : config.batch_size,
                      'num_workers': config.dataloader.val.num_workers,
                      'pin_memory' : False}

        train_params = {'batch_size' : config.batch_size,
                        'num_workers': config.dataloader.train.num_workers,
                        'shuffle':True,
                        'pin_memory' : False}
        train_loader = torch.utils.data.DataLoader(
            train_dataset, **train_params
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, **val_params
        )
        fbank_loader = torch.utils.data.DataLoader(
            feature_bank_dataset, **val_params
        )
        return train_loader, fbank_loader, valid_loader, train_dataset.classes
    elif config.dataset.name == 'CIFAR10':

        train_transform = SimSiamTransform(SIZE)
        # val_transform = SimSiamTransform(32)
        val_transform = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=root_dir, transform=train_transform, train=True,
                                                     download=True)

        feature_bank_dataset = torchvision.datasets.CIFAR10(root=root_dir, transform=train_transform, train=True,
                                                            download=True)
        valid_dataset = torchvision.datasets.CIFAR10(root=root_dir, transform=val_transform, train=False,
                                                     download=True)
        valid_size = 0.2
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_idx)
        #
        # valid_sampler = SubsetRandomSampler(valid_idx)
        val_params = {'batch_size' : config.batch_size,
                      'num_workers': config.dataloader.val.num_workers,
                      'pin_memory' : False}

        train_params = {'batch_size' : config.batch_size,
                        'num_workers': config.dataloader.train.num_workers,
                        'shuffle':True,
                        'pin_memory' : False}
        train_loader = torch.utils.data.DataLoader(
            train_dataset, **train_params
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, **val_params
        )
        fbank_loader = torch.utils.data.DataLoader(
            feature_bank_dataset, **val_params
        )
        return train_loader, fbank_loader, valid_loader, train_dataset.classes
    elif config.dataset.name == 'random':

        train_transform = SimSiamTransform(SIZE)
        dataset = RandomDataset(transform=train_transform)
        val_params = {'batch_size' : config.batch_size,
                      'num_workers': config.dataloader.val.num_workers,
                      'pin_memory' : False}

        train_params = {'batch_size' : config.batch_size,
                        'num_workers': config.dataloader.train.num_workers,
                        'pin_memory' : False}
        train_loader = torch.utils.data.DataLoader(
            dataset, **train_params
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset, **val_params
        )
        fbank_loader = torch.utils.data.DataLoader(
            dataset, **val_params
        )
        return train_loader, fbank_loader, valid_loader, dataset.classes
    elif config.dataset.name == 'STL10':

        train_transform = SimSiamTransform(SIZE)
        # val_transform = SimSiamTransform(32)
        val_transform = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.STL10(root=root_dir, transform=train_transform, split='train+unlabeled',
                                                   download=True)

        val_dataset = torchvision.datasets.STL10(root=root_dir, transform=train_transform, split='train',
                                                 download=True)
        test_dataset = torchvision.datasets.STL10(root=root_dir, transform=val_transform, split='test',
                                                  download=True)
        # valid_size = 0.2
        # num_train = len(train_dataset)
        # indices = list(range(num_train))
        # split = int(np.floor(valid_size * num_train))
        #
        # np.random.shuffle(indices)
        #
        # train_idx, valid_idx = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_idx)
        #
        # valid_sampler = SubsetRandomSampler(valid_idx)
        val_params = {'batch_size' : config.batch_size,
                      'num_workers': config.dataloader.val.num_workers,
                      'shuffle'    : False,
                      'pin_memory' : False}

        train_params = {'batch_size' : config.batch_size,
                        'num_workers': config.dataloader.train.num_workers,
                        'shuffle'    : True,
                        'pin_memory' : False}
        train_loader = torch.utils.data.DataLoader(
            train_dataset, **train_params
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset, **val_params
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **val_params
        )
        return train_loader, valid_loader, test_loader, train_dataset.classes
    elif config.dataset.name == 'celeba':
        train_transform = SimSiamTransform(SIZE)
        # val_transform = SimSiamTransform(32)
        val_transform = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # valid_sampler = SubsetRandomSampler(valid_idx)
        val_params = {'batch_size' : config.batch_size,
                      'num_workers': config.dataloader.val.num_workers,
                      'pin_memory' : False}

        train_params = {'batch_size' : config.batch_size,
                        'num_workers': config.dataloader.train.num_workers,
                        'shuffle':True,
                        'pin_memory' : False}
        train_dataset = torchvision.datasets.CelebA(root=root_dir, transform=train_transform, split='train',
                                                 download=True)
        val_dataset = torchvision.datasets.CelebA(root=root_dir, transform=train_transform, split='valid',
                                                 download=True)
        test_dataset = torchvision.datasets.CelebA(root=root_dir, transform=train_transform, split='test',
                                                 download=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, **train_params
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset, **val_params
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **val_params
        )
        print(train_dataset.identity.numpy())
        c = train_dataset.identity.numpy()
        print(len(c),len(np.unique(c)))
        return train_loader, valid_loader, test_loader, np.unique(c)
def get_dataset(dataset, data_dir, transform, train=True, download=True, debug_subset_size=None):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test',
                                             transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val',
                                                transform=transform, download=download)
    elif dataset == 'random':
        dataset = RandomDataset()
    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size))  # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    print(dataset.classes)
    return dataset
