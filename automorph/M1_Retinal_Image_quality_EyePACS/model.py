import argparse
import os
import numpy as np
import math
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets, models
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch



##########################  Unet ######################################3

def InceptionV3_fl(pretrained):
    inception_v3 = models.inception_v3(pretrained = True)
    inception_v3.fc = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 3)
        )
    inception_v3.fc = net_fl
    
    return inception_v3



def Efficientnet_fl(pretrained):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Identity()
    net_fl = nn.Sequential(
            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3)
            )
    model._fc = net_fl
    
    return model



def Densenet161_fl(pretrained):
    densenet161 = models.densenet161(pretrained = True)
    densenet161.classifier = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2208, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 3)
        )
    densenet161.classifier = net_fl
    
    return densenet161


def Resnet101_fl(pretrained):
    resnet101 = models.resnet101(pretrained = True)
    resnet101.fc = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 3)
        )
    resnet101.fc = net_fl
    
    return resnet101


def Resnext101_32x8d_fl(pretrained):
    resnext101_32x8d = models.resnext101_32x8d(pretrained = True)
    resnext101_32x8d.fc = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 3)
        )
    resnext101_32x8d.fc = net_fl
    
    return resnext101_32x8d


def MobilenetV2_fl(pretrained):
    mobilenet_v2 = models.mobilenet_v2(pretrained = True)
    mobilenet_v2.classifier = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 3)
        )
    mobilenet_v2.classifier = net_fl
    
    return mobilenet_v2



def Vgg16_bn_fl(pretrained):
    vgg16_bn = models.vgg16_bn(pretrained = True)
    vgg16_bn.classifier = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 3)
        )
    vgg16_bn.classifier = net_fl
    
    return vgg16_bn


