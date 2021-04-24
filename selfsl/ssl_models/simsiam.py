import timm
# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab
# /moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        #print(x.shape)
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        #print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()

        self.backbone, in_feats = select_backbone(backbone, pretrained=False)
        self.projector = projection_MLP(in_feats)

        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        #print(x1.shape,x2.shape)
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L


def select_backbone(model, pretrained=False):
    print(model)
    if (model == 'resnet50'):
        cnn = models.resnet50(pretrained=pretrained)
        cnn.fc = nn.Identity()
        in_feats = 2048
    elif (model == 'resnet18'):
        cnn = models.resnet18(pretrained=pretrained)
        cnn.fc = nn.Identity()
        in_feats = 512
    elif (model == 'resnext50_32x4d'):

        cnn = models.resnext50_32x4d(pretrained=pretrained)

    elif (model == 'mobilenet_v2'):

        cnn = models.mobilenet_v2(pretrained=pretrained)

    elif (model == 'densenet121'):
        cnn = models.densenet121(pretrained=pretrained)
        in_feats = 1024

    elif model == 'efficientnet_b1':

        cnn = timm.create_model('efficientnet_b1', pretrained=pretrained)
        in_feats = 1280
    #cnn = nn.Sequential(*list(cnn.children())[:-1])  # do not return classifier
    return cnn, in_feats


def knn_monitor(net, val_data_loader, test_data_loader, epoch, logger, k=200, t=0.1):
    net.eval()
    classes = len(val_data_loader.dataset.classes)
    print(classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for (data,data2), target in val_data_loader:
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(val_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search

        if test_data_loader is not None:
            for data, target in test_data_loader:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data)
                feature = F.normalize(feature, dim=1)

                pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            logger.info(f'Accuracy {total_top1 / total_num * 100:.5f} %')
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
