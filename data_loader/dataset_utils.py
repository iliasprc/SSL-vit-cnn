
from data_loader.cifar100 import cifar100
def select_dataset(config):
    test_params = {'batch_size': config.dataloader.test.batch_size,
                   'shuffle': False,
                   'num_workers': 2}
    val_params = {'batch_size': config.dataloader.val.batch_size,
                  'shuffle': config.dataloader.val.shuffle,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory': True}

    train_params = {'batch_size': config.dataloader.train.batch_size,
                    'shuffle': config.dataloader.train.shuffle,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory': True}
    if config.dataset.name == 'CIFAR100':
        return cifar100(config)

