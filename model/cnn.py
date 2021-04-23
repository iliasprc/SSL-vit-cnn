import torch.nn as nn
import timm
from torchvision import models
import timm

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

    def forward(self, x):

        return self.cnn(x)
