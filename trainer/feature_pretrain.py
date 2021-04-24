import torch
from byol_pytorch import BYOL

from model import  ViT

n_classes = 3



def sample_unlabelled_images(size=224):
    return torch.randn(32, 3, size, size)



def byol_pretrain(model, size=224):
    learner = BYOL(
        model,
        image_size=size,
        hidden_layer='fc2'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=1e-4)
    for idx in range(50):
        images = sample_unlabelled_images(size)
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        if idx%10==0:
            print(loss.item())
        opt.step()
        learner.update_moving_average()  # update moving average of target encoder
    return model
