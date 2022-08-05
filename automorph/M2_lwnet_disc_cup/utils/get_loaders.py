from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from . import paired_transforms_tv04 as p_tr
from os.path import splitext
from os import listdir
import os
import os.path as osp
import pandas as pd
from PIL import Image
import numpy as np
from skimage.measure import regionprops
import torch
import logging
from glob import glob

class TrainDataset(Dataset):
    def __init__(self, csv_path, transforms=None, label_values=None):
        
        self.im_list = csv_path + 'images/'
        self.gt_list = csv_path + '1st_manual/'
        #self.mask_list = df.mask_paths
        self.transforms = transforms
        self.label_values = label_values  # for use in label_encoding
        
        self.ids = [splitext(file)[0] for file in listdir(self.im_list)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {(self.ids)} ')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        tg_crop = Image.fromarray(np.array(target)[minr:maxr, minc:maxc])
        mask_crop = Image.fromarray(np.array(mask)[minr:maxr, minc:maxc])
        return im_crop, tg_crop, mask_crop

    def __getitem__(self, index):
        # load image and labels
        idx = self.ids[index]
        
        label_file = glob(self.gt_list + idx  + '.*')
        img_file = glob(self.im_list + idx + '.*')
        
        img = Image.open(img_file[0])
        target = Image.open(label_file[0])
        #mask = Image.open(self.mask_list[index]).convert('L')

        #img, target, mask = self.crop_to_fov(img, target, mask)

        target = self.label_encoding(target)

        target = np.array(self.label_encoding(target))

        #target[np.array(mask) == 0] = 0
        target = Image.fromarray(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        # QUICK HACK FOR PSEUDO_SEG IN VESSELS, BUT IT SPOILS A/V
        if len(self.label_values)==2: # vessel segmentation case
            target = target.float()
            if torch.max(target) >1:
                target= target.float()/255

        return img, target

    def __len__(self):
        return len(self.ids)

class TestDataset(Dataset):
    def __init__(self, crop_csv, csv_path, tg_size):
        
        self.im_list = csv_path
        self.crop_csv = crop_csv
        fps = pd.read_csv(crop_csv, usecols=['Name']).values.ravel()
        self.file_paths = fps
        logging.info(f'Creating dataset with {(self.file_paths)} ')
        logging.info(f'Creating dataset with {len(self.file_paths)} examples')
        
        #self.mask_list = df.mask_paths
        self.tg_size = tg_size

    def crop_to_fov(self, img, mask):
        mask = np.array(mask).astype(int)
        minr, minc, maxr, maxc = regionprops(mask)[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]

    @classmethod
    def crop_img(self, crop_csv, f_path, pil_img):
        """ Code to crop the input image based on the crops stored in a csv. This is done to save space and having to store intermediate cropped
        files.
        Params:
        crop_csv - csv containing the name with filepath stored at gv.image_dir, and crop info
        f_path - str containing the filepath to the image
        pil_img - PIL Image of the above f_path
        Returns:
        pil_img - PIL Image cropped by the data in the csv
        """ 

        df = pd.read_csv(crop_csv)
        row = df[df['Name'] == f_path]
        
        c_w = row['centre_w']
        c_h = row['centre_h']
        r = row['radius']
        w_min, w_max = int(c_w-r), int(c_w+r) 
        h_min, h_max = int(c_h-r), int(c_h+r)
        
        pil_img = pil_img.crop((h_min, w_min, h_max, w_max))

        return pil_img
 

    def __getitem__(self, index):
        # # load image and mask
        img_file = self.file_paths[index]

        img = Image.open(img_file)
        img = self.crop_img(self.crop_csv, img_file, img)
        
        #mask = Image.open(self.mask_list[index]).convert('L')
        #img, coords_crop = self.crop_to_fov(img, mask)
        original_sz = img.size[0], img.size[1]  # in numpy convention

        # # load image and mask
        # img = Image.open(self.im_list[index])
        # original_sz = img.size[1], img.size[0]  # in numpy convention
        # mask = Image.open(self.mask_list[index]).convert('L')
        # img, coords_crop = self.crop_to_fov(img, mask)
        # print(self.im_list[index], 'original size inside dataset', original_sz)

        rsz = p_tr.Resize(self.tg_size)
        tnsr = p_tr.ToTensor()
        tr = p_tr.Compose([rsz, tnsr])
        img = tr(img)  # only transform image

        return {
            'name': img_file.split('/')[-1].split('.')[0],
            'image': img,
            'original_sz': original_sz
        }

    def __len__(self):
        return len(self.file_paths)

    

def build_pseudo_dataset(train_csv_path, test_csv_path, path_to_preds):
    # assumes predictions are in path_to_preds and have the same name as images in the test csv
    # image extension does not matter
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # If there are more pseudo-segmentations than training segmentations
    # we bootstrap training images to get same numbers
    missing = test_df.shape[0] - train_df.shape[0]
    if missing > 0:
        extra_segs = train_df.sample(n=missing, replace=True, random_state=42)
        train_df = pd.concat([train_df, extra_segs])


    train_im_list = list(train_df.im_paths)
    train_gt_list = list(train_df.gt_paths)
    train_mask_list = list(train_df.mask_paths)

    test_im_list = list(test_df.im_paths)
    test_mask_list = list(test_df.mask_paths)

    test_preds = [n for n in os.listdir(path_to_preds) if 'binary' not in n and 'perf' not in n]
    test_pseudo_gt_list = []

    for n in test_im_list:
        im_name_no_extension = n.split('/')[-1][:-4]
        for pred_name in test_preds:
            pred_name_no_extension = pred_name.split('/')[-1][:-4]
            if im_name_no_extension == pred_name_no_extension:
                test_pseudo_gt_list.append(osp.join(path_to_preds, pred_name))
                break
    train_im_list.extend(test_im_list)
    train_gt_list.extend(test_pseudo_gt_list)
    train_mask_list.extend(test_mask_list)
    return train_im_list, train_gt_list, train_mask_list


def get_train_val_datasets(csv_path_train, csv_path_val, seed_num, tg_size=(512, 512), label_values=(0, 255)):

    
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=45, fill=(0, 0, 0), fill_tg=(0,))
    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = p_tr.Compose([resize,  scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    val_transforms = p_tr.Compose([resize, tensorizer])
    #train_dataset.transforms = train_transforms
    #val_dataset.transforms = val_transforms
    
    
    train_dataset_all = TrainDataset(csv_path=csv_path_train, transforms=train_transforms, label_values=label_values)
    n_val = int(len(train_dataset_all) * 0.2)
    n_train = len(train_dataset_all) - n_val
    torch.manual_seed(seed_num)
    train_dataset, val_dataset = random_split(train_dataset_all, [n_train, n_val])
    
    
    
    #val_dataset = TrainDataset(csv_path=csv_path_val, label_values=label_values)
    # transforms definition
    # required transforms


    return train_dataset, val_dataset

def get_train_val_loaders(csv_path_train, csv_path_val, seed_num, batch_size=4, tg_size=(512, 512), label_values=(0, 255), num_workers=0):
    train_dataset, val_dataset = get_train_val_datasets(csv_path_train, csv_path_val, seed_num, tg_size=tg_size, label_values=label_values)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader

def get_test_dataset(data_path, crop_csv, csv_path='test.csv', tg_size=(512, 512), batch_size=1, num_workers=1):
    # csv_path will only not be test.csv when we want to build training set predictions
    #path_test_csv = osp.join(data_path, csv_path)
    path_test_csv = data_path
    test_dataset = TestDataset(crop_csv, csv_path=path_test_csv, tg_size=tg_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    return test_loader



