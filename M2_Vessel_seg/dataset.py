from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms, utils
import random
from torchvision.utils import save_image
from scipy.ndimage import rotate
from PIL import Image, ImageEnhance
import cv2


class SEDataset(Dataset):
    def __init__(self, imgs_dir, label_dir, mask_dir, img_size, dataset_name, pthrehold, uniform, train_or=True):
        self.imgs_dir = imgs_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.dataset_name = dataset_name
        self.pthrehold = pthrehold
        self.uniform = uniform
        self.train_or = train_or

        
        i = 0
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #logging.info(f'Creating dataset with {(self.ids)} ')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def pad_imgs(self, imgs, img_size):
        img_h,img_w=imgs.shape[0], imgs.shape[1]
        target_h,target_w=img_size[0],img_size[1] 
        if len(imgs.shape)==3:
            d=imgs.shape[2]
            padded=np.zeros((target_h, target_w,d))
        elif len(imgs.shape)==2:
            padded=np.zeros((target_h, target_w))
        padded[(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=imgs
        return padded

    @classmethod
    def random_perturbation(self,imgs):
        for i in range(imgs.shape[0]):
            im=Image.fromarray(imgs[i,...].astype(np.uint8))
            en=ImageEnhance.Color(im)
            im=en.enhance(random.uniform(0.8,1.2))
            imgs[i,...]= np.asarray(im).astype(np.float32)
        return imgs 

    @classmethod
    def preprocess(self, img, label, mask, dataset_name, img_size, train_or, pthrehold):

        img_array = np.array(img)
        label_array = np.array(label).astype(np.float32)
        mask_array = np.array(mask).astype(np.float32)
        vessel_max = np.amax(label_array)
        mask_max = np.amax(mask_array)
        
        if mask_array.ndim>2:
            mask_array = mask_array[...,0]
        if vessel_max>1:
            label_array = label_array/255.0
        if mask_max>1:
            mask_array = mask_array/255.0
            
        if dataset_name=='STARE' or dataset_name=='DRIVE':
            img_array = self.pad_imgs(img_array, img_size)
            if train_or:
                label_array = self.pad_imgs(label_array, img_size)
                mask_array = self.pad_imgs(mask_array, img_size)
        
        if train_or:
            if np.random.random()>0.5:
                img_array=img_array[:,::-1,:]    # flipped imgs
                label_array=label_array[:,::-1]
                mask_array=mask_array[:,::-1]

            angle = np.random.randint(360)
            img_array = rotate(img_array, angle, axes=(0, 1), reshape=False)
            img_array = self.random_perturbation(img_array)
            label_array = np.round(rotate(label_array, angle, axes=(0, 1), reshape=False))
            mask_array = np.round(rotate(mask_array, angle, axes=(0, 1), reshape=False))
        
        if dataset_name=='WIDE':
            img_array = np.concatenate((img_array[...,1][...,np.newaxis],img_array[...,1][...,np.newaxis],img_array[...,1][...,np.newaxis]),axis=2)
            mean_value=np.mean(img_array[img_array[...,0] > pthrehold],axis=0)
            std_value=np.std(img_array[img_array[...,0] > pthrehold],axis=0)
            img_array=(img_array-mean_value)/std_value
        else:
            mean_value=np.mean(img_array[img_array[...,0] > pthrehold],axis=0)
            std_value=np.std(img_array[img_array[...,0] > pthrehold],axis=0)
            img_array=(img_array-mean_value)/std_value
        
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        if len(label_array.shape) == 2:
            label_array = np.expand_dims(label_array, axis=2)
        
        
        img_array = img_array.transpose((2, 0, 1))
        label_array = label_array.transpose((2, 0, 1))
        #mask_array = mask_array.transpose((2, 0, 1))

        label_array = np.where(label_array > 0.5, 1, 0)
        mask_array = np.where(mask_array > 0.5, 1, 0)

        return img_array, label_array, mask_array


    def __getitem__(self, i):
        idx = self.ids[i]

        if self.dataset_name=='CHASEDB1':
            mask_file = glob(self.label_dir + idx + '_1stHO'  + '.*')
            img_file = glob(self.imgs_dir + idx + '.*')
            module_file = glob(self.mask_dir + idx + '_MASK' + '.*')
        
        elif self.dataset_name=='STARE':
            mask_file = glob(self.label_dir + idx + '.ah'  + '.*')
            img_file = glob(self.imgs_dir + idx + '.*')
            module_file = glob(self.mask_dir + idx + '.*')

        elif self.dataset_name=='IOSTAR':
            mask_file = glob(self.label_dir + idx + '_GT'  + '.*')
            img_file = glob(self.imgs_dir + idx + '.*')
            module_file = glob(self.mask_dir + idx + '_Mask' + '.*')
        
        elif self.dataset_name=='WIDE':
            mask_file = glob(self.label_dir + idx + '_vessels'  + '.*')
            img_file = glob(self.imgs_dir + idx + '.*')
            module_file = glob(self.mask_dir + idx + '_vessels' + '.*')
            
        elif self.dataset_name=='DR-HAGIS':
            mask_file = glob(self.label_dir + idx + '.*')
            img_file = glob(self.imgs_dir + idx + '.*')
            module_file = glob(self.mask_dir + idx + '.*')
        
        else:
            mask_file = glob(self.label_dir + idx  + '.*')
            img_file = glob(self.imgs_dir + idx + '.*')
            module_file = glob(self.mask_dir + idx + '_mask' + '.*')

        

        if self.train_or:
            label = Image.open(mask_file[0]).resize(self.img_size)
            img = Image.open(img_file[0]).resize(self.img_size)
            mask = Image.open(module_file[0]).resize(self.img_size)
        
        else:
            if self.dataset_name!='STARE' or self.dataset_name!='DRIVE':
                label = Image.open(mask_file[0])
                img = Image.open(img_file[0]).resize(self.img_size)
                mask = Image.open(module_file[0])
            else:
                label = Image.open(mask_file[0])
                img = Image.open(img_file[0])
                mask = Image.open(module_file[0])
        
        img, label, mask = self.preprocess(img, label, mask, self.dataset_name, self.img_size, self.train_or, self.pthrehold)

        i += 1
        
        return {
            'name': idx,
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor),
            'mask':torch.from_numpy(mask).type(torch.FloatTensor)
        }



    
    
    
    
    
class SEDataset_out(Dataset):
    def __init__(self, imgs_dir, label_dir, mask_dir, img_size, dataset_name, pthrehold, uniform, train_or=True):
        self.imgs_dir = imgs_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.dataset_name = dataset_name
        self.pthrehold = pthrehold
        self.uniform = uniform
        self.train_or = train_or

        
        i = 0
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #logging.info(f'Creating dataset with {(self.ids)} ')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def pad_imgs(self, imgs, img_size):
        img_h,img_w=imgs.shape[0], imgs.shape[1]
        target_h,target_w=img_size[0],img_size[1] 
        if len(imgs.shape)==3:
            d=imgs.shape[2]
            padded=np.zeros((target_h, target_w,d))
        elif len(imgs.shape)==2:
            padded=np.zeros((target_h, target_w))
        padded[(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=imgs
        return padded

    @classmethod
    def random_perturbation(self,imgs):
        for i in range(imgs.shape[0]):
            im=Image.fromarray(imgs[i,...].astype(np.uint8))
            en=ImageEnhance.Color(im)
            im=en.enhance(random.uniform(0.8,1.2))
            imgs[i,...]= np.asarray(im).astype(np.float32)
        return imgs 

    @classmethod
    def preprocess(self, img, dataset_name, img_size, train_or, pthrehold):

        img_array = np.array(img)

        if np.sum(img_array[...,2])==0:
            img_array = np.concatenate((img_array[...,1][...,np.newaxis],img_array[...,1][...,np.newaxis],img_array[...,1][...,np.newaxis]),axis=2)
            mean_value=np.mean(img_array[img_array[...,0] > pthrehold],axis=0)
            std_value=np.std(img_array[img_array[...,0] > pthrehold],axis=0)
            img_array=(img_array-mean_value)/std_value        
            
        else:
            mean_value=np.mean(img_array[img_array[...,0] > pthrehold],axis=0)
            std_value=np.std(img_array[img_array[...,0] > pthrehold],axis=0)
            img_array=(img_array-mean_value)/std_value

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
            

        
        img_array = img_array.transpose((2, 0, 1))




        return img_array


    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        img = Image.open(img_file[0])
        ori_width, ori_height = img.size
        img = img.resize(self.img_size)
        img = self.preprocess(img, self.dataset_name, self.img_size, self.train_or, self.pthrehold)

        i += 1
        
        return {
            'name': idx,
            'width': ori_width,
            'height': ori_height,
            'image': torch.from_numpy(img).type(torch.FloatTensor)
        }

