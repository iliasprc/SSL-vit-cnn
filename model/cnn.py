import torch.nn as nn
import timm
from torchvision import models
import timm

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
        if (model == 'resnet50'):
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
        elif model =='efficientnet_b1':

            self.cnn  = timm.create_model('efficientnet_b1', pretrained=True,num_classes=num_classes)
        self.cnn = nn.Sequential(*list(self.cnn.children()))

    def forward(self, x):

        return self.cnn(x)



class ByolCNN(nn.Module):
    def __init__(self, **kwargs):
        super(ByolCNN, self).__init__()
        pretrained =  kwargs['pretrained']
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
        elif model =='efficientnet_b0':

            self.cnn  = timm.create_model('efficientnet_b0', pretrained=pretrained)
        elif model =='efficientnet_b1':

            self.cnn  = timm.create_model('efficientnet_b1', pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.projetion = MLPHead(in_channels=in_feats , **kwargs['projection_head'])

    def forward(self, x):

        return self.cnn(x)
