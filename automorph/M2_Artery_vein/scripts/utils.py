import os
import sys
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def Define_image_size(uniform, dataset):
    
    if uniform =='True':
        img_size = (720,720)
    else:
        if dataset=='HRF-AV':
            img_size = (880,592)  
        elif dataset=='DRIVE_AV':
            img_size = (592,592)
        elif dataset=='LES-AV':
            img_size = (800,720)
            
    return img_size

