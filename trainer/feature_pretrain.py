import torch
from byol_pytorch import BYOL

from model import CovidNet, ViT

n_classes = 3

model = CovidNet('small', n_classes=n_classes)

# model = CovidNet('large', n_classes=n_classes)
#
# model = CNN(n_classes, 'mobilenet_v2')
#
model = ViT(
    image_size=224,
    patch_size=32,
    num_classes=3,
    dim=512,
    depth=6,
    heads=16,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)
print(model)


def sample_unlabelled_images(size=224):
    return torch.randn(32, 3, size, size)


for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    print(loss.item())
    opt.step()
    learner.update_moving_average()  # update moving average of target encoder

# save your improved network
torch.save(model.state_dict(), './improved-covidnet_small.pt')


def byol_pretrain(model, size=224):
    learner = BYOL(
        model,
        image_size=size,
        hidden_layer='fc2'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=1e-4)
    for _ in range(100):
        images = sample_unlabelled_images(size)
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        print(loss.item())
        opt.step()
        learner.update_moving_average()  # update moving average of target encoder
    return model
