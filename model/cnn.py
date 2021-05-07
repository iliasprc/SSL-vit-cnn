import timm
import torch.nn as nn
from torchvision import models

from model.mlp_head import MLPHead


def cnn_select(num_classes, model='resnet50', pretrained=False):
    if (model == 'resnet50'):
        cnn = models.resnet50(pretrained=pretrained)
        cnn.fc = nn.Linear(2048, num_classes)
    elif (model == 'resnext50_32x4d'):

        cnn = models.resnext50_32x4d(pretrained=pretrained)
        cnn.classifier = nn.Linear(1280, num_classes)
    elif (model == 'mobilenet_v2'):

        cnn = models.mobilenet_v2(pretrained=pretrained)
        cnn.classifier = nn.Linear(1280, num_classes)
    elif (model == 'densenet121'):
        cnn = models.densenet121(pretrained=pretrained)
        cnn.classifier = nn.Linear(1024, num_classes)
    elif model == 'efficientnet_b1':

        cnn = timm.create_model('efficientnet_b1', pretrained=True, num_classes=num_classes)

    return cnn


class CNN(nn.Module):
    def __init__(self, num_classes, model='resnet50', pretrained=False):
        super(CNN, self).__init__()
        if (model == 'resnet18'):
            self.cnn = models.resnet18(pretrained=pretrained)
            self.cnn.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.cnn.fc = nn.Linear(512, num_classes)
        elif (model == 'resnet50'):
            self.cnn = models.resnet50(pretrained=pretrained)
            self.cnn.fc = nn.Linear(2048, num_classes)
        elif (model == 'resnext50_32x4d'):

            self.cnn = models.resnext50_32x4d(pretrained=pretrained)
            self.cnn.classifier = nn.Linear(1280, num_classes)
        elif (model == 'mobilenet_v2'):

            self.cnn = models.mobilenet_v2(pretrained=pretrained)
            self.cnn.classifier = nn.Linear(1280, num_classes)
        elif (model == 'densenet121'):
            self.cnn = models.densenet121(pretrained=pretrained)
            self.cnn.classifier = nn.Linear(1024, num_classes)
        elif model == 'efficientnet_b1':

            self.cnn = timm.create_model('efficientnet_b1', pretrained=True, num_classes=num_classes)
        # self.cnn = nn.Sequential(*list(self.cnn.children()))
        elif model == 'efficientnet_b0':

            self.cnn = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)

            #self.cnn.conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        elif model == 'vit':
            patch_size = 8

            patch_size = 16
            embed_dim = 768
            # cnn =  ViT(
            #     image_size = shape,
            #     patch_size = patch_size,
            #     num_classes = 1000,
            #     dim = embed_dim,
            #     depth = 3,
            #     heads = 8,
            #     mlp_dim = 2*embed_dim,
            #     dropout = 0.2,
            #     emb_dropout = 0.2
            # )
            # cnn.mlp_head = nn.Identity()
            self.cnn = timm.create_model('vit_small_patch16_224', pretrained=pretrained, img_size=224, num_classes=num_classes)
            in_feats = embed_dim
            #
            #self.cnn.head = nn.Identity()
        elif model == 'pit':
            self.cnn = timm.create_model('pit_ti_224', pretrained=pretrained,img_size=224, num_classes=num_classes)
    def forward(self, x):

        return self.cnn(x)


class ByolCNN(nn.Module):
    def __init__(self, **kwargs):
        super(ByolCNN, self).__init__()
        pretrained = kwargs['pretrained']
        model = kwargs['model']
        if (model == 'resnet50'):
            self.cnn = models.resnet50(pretrained=pretrained)
            in_feats = 2048

        elif (model == 'resnext50_32x4d'):

            self.cnn = models.resnext50_32x4d(pretrained=pretrained)

        elif (model == 'mobilenet_v2'):

            self.cnn = models.mobilenet_v2(pretrained=pretrained)

        elif (model == 'densenet121'):
            self.cnn = models.densenet121(pretrained=pretrained)
        elif model == 'efficientnet_b0':

            self.cnn = timm.create_model('efficientnet_b0', pretrained=pretrained)
        elif model == 'efficientnet_b1':

            self.cnn = timm.create_model('efficientnet_b1', pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.projetion = MLPHead(in_channels=in_feats, **kwargs['projection_head'])

    def forward(self, x):

        return self.cnn(x)
