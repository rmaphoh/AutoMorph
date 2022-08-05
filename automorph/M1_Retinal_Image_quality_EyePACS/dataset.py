import os
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, utils
import random
import logging
from scipy.ndimage import rotate
from PIL import Image, ImageEnhance
import pandas as pd
from os.path import splitext
from os import listdir


class BasicDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, image_base, list_IDs, image_size, n_classes, train_or):
        'Initialization'
        self.image_size = image_size
        self.image_base = image_base
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.train_or = train_or
        df = pd.read_csv(self.list_IDs)
        self.list_IDs = df.values.tolist()
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


    @classmethod
    def random_perturbation(self,imgs,top,bot):

        im=Image.fromarray(imgs.astype(np.uint8))
        en=ImageEnhance.Color(im)
        im=en.enhance(random.uniform(bot,top))
        imgs= np.asarray(im).astype(np.float32)
        
        return imgs 

    @classmethod
    def preprocess(self, pil_img, img_size, train_or,index):
        #w, h = pil_img.size
        newW, newH = img_size[0], img_size[1]
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_array = np.array(pil_img)
        
        if train_or:
            if np.random.random()>0.5:
                img_array=img_array[:,::-1,:]    # flipped imgs
            if np.random.random()>0.5:
                img_array=img_array[::-1,:,:]    # flipped imgs            
            angle = np.random.randint(360)
            img_array = rotate(img_array, angle, axes=(1, 0), reshape=False)
            img_array = self.random_perturbation(img_array, top=1.2, bot=0.8)

        mean_value=np.mean(img_array[img_array > 0])
        std_value=np.std(img_array[img_array > 0])
        img_array=(img_array-mean_value)/std_value
        #print(np.unique(img_array))
        
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
            img_array = np.concatenate((img_array,img_array,img_array),axis=2)
        
        img_array = img_array.transpose((2, 0, 1))
        
        return img_array
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        _,img_name,label,_ = self.list_IDs[index]
        
        img_name = self.image_base + img_name.split('.')[0] + '.png'
        image = Image.open(img_name)
        image_processed = self.preprocess(image, self.image_size, self.train_or, index)
 
        return {
            'img_file': img_name,
            'image': torch.from_numpy(image_processed).type(torch.FloatTensor),
            'label': torch.from_numpy(np.eye(self.n_classes, dtype='uint8')[label]).type(torch.long)
        }




class BasicDataset_OUT(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, image_dir, image_size, n_classes, train_or, crop_csv):
        'Initialization'
        self.image_size = image_size
        self.image_dir = image_dir
        self.n_classes = n_classes
        self.train_or = train_or
        self.crop_csv = crop_csv
        fps = pd.read_csv(crop_csv, usecols=['Name']).values.flatten()
        self.file_paths = fps

        # create the dataset by accessing the crop info and loading the 

        logging.info(f'Creating dataset with {len(self.file_paths)} examples')

        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_paths)

    @classmethod
    def preprocess(self, pil_img, img_size, train_or, img_path, crop_csv):
        #w, h = pil_img.size

        # crop the image based on the pre-poc csv at M0
        df = pd.read_csv(crop_csv)
        row = df[df['Name'] == img_path]
        
        c_w = row['centre_w']
        c_h = row['centre_h']
        r = row['radius']
        w_min, w_max = int(c_w-r), int(c_w+r) 
        h_min, h_max = int(c_h-r), int(c_h+r)
        
        pil_img = pil_img.crop((h_min, w_min, h_max, w_max))
        #pil_img.save("/data/anand/test_cropped.png") #TODO remove

        newW, newH = img_size[0], img_size[1]
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_array = np.array(pil_img)
        
        mean_value=np.mean(img_array[img_array > 0])
        std_value=np.std(img_array[img_array > 0])
        img_array=(img_array-mean_value)/std_value
        
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
            img_array = np.concatenate((img_array,img_array,img_array),axis=2)
        
        img_array = img_array.transpose((2, 0, 1))
        
        return img_array

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_file = self.file_paths[index]
        image = Image.open(img_file)
        image_processed = self.preprocess(image, self.image_size, self.train_or, img_file, self.crop_csv)
 
        return {
            'img_file': img_file,
            'image': torch.from_numpy(image_processed).type(torch.FloatTensor)
        }


