import torch
from PIL import Image
import numpy as np
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, img_size=224,root=None, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.size = 1000
        self.classes = ['0']
    def __getitem__(self, idx):
        if idx < self.size:
            img = Image.fromarray((255.0*torch.randn((224, 224,3)).numpy()).astype(np.uint8))

            return self.transform(img), [0,0,0]
        else:
            raise Exception

    def __len__(self):
        return self.size