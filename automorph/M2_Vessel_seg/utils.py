import os
import sys
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def Define_image_size(uniform, dataset):
    
    if uniform =='True':
        img_size = (912,912)
            
    return img_size

