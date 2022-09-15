'''
yukun 20210801
'''

import torch.nn.functional as F
import argparse
import logging
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .model import Discriminator, Segmenter
from .dataset import SEDataset_out
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from .utils import Define_image_size
import torchvision
from skimage.morphology import skeletonize,remove_small_objects
from skimage import io
from .FD_cal import fractal_dimension,vessel_density
import shutil
from pathlib import Path


def filter_frag(data_path):
    if os.path.isdir(data_path + 'resize_binary/.ipynb_checkpoints'):
        shutil.rmtree(data_path + 'resize_binary/.ipynb_checkpoints')

    image_list=os.listdir(data_path + 'resize_binary')
    FD_cal=[]
    name_list=[]
    VD_cal=[]
    width_cal=[]

    for i in sorted(image_list):
        img=io.imread(data_path + 'resize_binary/' + i, as_gray=True).astype(np.int64)
        img2=img>0
        img2 = remove_small_objects(img2, 30, connectivity=5)
        
        if not os.path.isdir(data_path + 'binary_process/'):
            os.makedirs(data_path + 'binary_process/') 
        io.imsave(data_path + 'binary_process/' + i , 255*(img2.astype('uint8')),check_contrast=False)

        skeleton = skeletonize(img2)
        
        if not os.path.isdir(data_path + 'binary_skeleton/'):
            os.makedirs(data_path + 'binary_skeleton/') 
        io.imsave(data_path + 'binary_skeleton/' + i, 255*(skeleton.astype('uint8')),check_contrast=False)
        
        FD_boxcounting = fractal_dimension(img2)
        VD = vessel_density(img2)
        width = np.sum(img2)/np.sum(skeleton)
        FD_cal.append(FD_boxcounting)
        name_list.append(i)
        VD_cal.append(VD)
        width_cal.append(width)
    
    return FD_cal,name_list,VD_cal,width_cal


def segment_fundus(data_path, net_1, net_2, net_3, net_4, net_5, net_6, net_7, net_8, net_9, net_10, loader, device, dataset_name, job_name, mask_or, train_or):

    n_val = len(loader) 
    tot = 0
    num = 0
    i = 0
    
    seg_results_small_path='./outside_test/segs/'
    seg_results_raw_path='./outside_test/segs/'
    seg_results_small_path = data_path + 'resize/'
    seg_results_small_binary_path = data_path + 'resize_binary/'
    seg_results_raw_path = data_path + 'raw/'
    seg_results_raw_binary_path = data_path + 'raw_binary/'
    
    

    if not os.path.isdir(seg_results_small_path):
        os.makedirs(seg_results_small_path)

    if not os.path.isdir(seg_results_raw_path):
        os.makedirs(seg_results_raw_path)
        
    if not os.path.isdir(seg_results_small_binary_path):
        os.makedirs(seg_results_small_binary_path)

    if not os.path.isdir(seg_results_raw_binary_path):
        os.makedirs(seg_results_raw_binary_path)
    
    
    seg_uncertainty_small_path = data_path + 'resize_uncertainty/'        
    if not os.path.isdir(seg_uncertainty_small_path):
        os.makedirs(seg_uncertainty_small_path)
    
    seg_uncertainty_raw_path = data_path + 'raw_uncertainty/'
    
    if not os.path.isdir(seg_uncertainty_raw_path):
        os.makedirs(seg_uncertainty_raw_path)
        
        
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            ori_width=batch['width']
            ori_height=batch['height']
            #img_name = batch['name'][0]
            img_name = batch['name']
            
            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred_1 = net_1(imgs)
                mask_pred_2 = net_2(imgs)
                mask_pred_3 = net_3(imgs)
                mask_pred_4 = net_4(imgs)
                mask_pred_5 = net_5(imgs)
                mask_pred_6 = net_6(imgs)
                mask_pred_7 = net_7(imgs)
                mask_pred_8 = net_8(imgs)
                mask_pred_9 = net_9(imgs)
                mask_pred_10 = net_10(imgs)
                
            mask_pred_sigmoid_1 = torch.sigmoid(mask_pred_1)
            mask_pred_sigmoid_2 = torch.sigmoid(mask_pred_2)
            mask_pred_sigmoid_3 = torch.sigmoid(mask_pred_3)
            mask_pred_sigmoid_4 = torch.sigmoid(mask_pred_4)
            mask_pred_sigmoid_5 = torch.sigmoid(mask_pred_5)
            mask_pred_sigmoid_6 = torch.sigmoid(mask_pred_6)
            mask_pred_sigmoid_7 = torch.sigmoid(mask_pred_7)
            mask_pred_sigmoid_8 = torch.sigmoid(mask_pred_8)
            mask_pred_sigmoid_9 = torch.sigmoid(mask_pred_9)
            mask_pred_sigmoid_10 = torch.sigmoid(mask_pred_10)
            
            mask_pred_sigmoid=(mask_pred_sigmoid_1+mask_pred_sigmoid_2+mask_pred_sigmoid_3+mask_pred_sigmoid_4+mask_pred_sigmoid_5+mask_pred_sigmoid_6+mask_pred_sigmoid_7+mask_pred_sigmoid_8+mask_pred_sigmoid_9+mask_pred_sigmoid_10)/10
            
            uncertainty_map = torch.sqrt((torch.square(mask_pred_sigmoid-mask_pred_sigmoid_1)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_2)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_3)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_4)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_5)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_6)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_7)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_8)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_9)+torch.square(mask_pred_sigmoid-mask_pred_sigmoid_10))/10)
            
            
            n_image = mask_pred_sigmoid.shape[0]
            
            for i in range(n_image):
                
                n_img_name = img_name[i]
                n_ori_width = ori_width[i]
                n_ori_height = ori_height[i]

                save_image(torch.unsqueeze(uncertainty_map[i,...], 0), seg_uncertainty_small_path+n_img_name+'.png')
                uncertainty_img = Image.open(seg_uncertainty_small_path+n_img_name+'.png').resize((n_ori_width,n_ori_height)).convert('L') 
                uncertainty_tensor = torchvision.transforms.ToTensor()(uncertainty_img)
                save_image(uncertainty_tensor, seg_uncertainty_raw_path+n_img_name+'.png')

                save_image(torch.unsqueeze(mask_pred_sigmoid[i,...], 0), seg_results_small_path+n_img_name+'.png')
                mask_pred_resize_bin=torch.zeros(torch.unsqueeze(mask_pred_sigmoid[i,...], 0).shape)
                mask_pred_resize_bin[torch.unsqueeze(mask_pred_sigmoid[i,...], 0)>=0.5]=1
                save_image(mask_pred_resize_bin, seg_results_small_binary_path+n_img_name+'.png')

                mask_pred_img = Image.open(seg_results_small_path+n_img_name+'.png').resize((n_ori_width,n_ori_height)).convert('L') 
                mask_pred_tensor = torchvision.transforms.ToTensor()(mask_pred_img)

                mask_pred_numpy_bin=torch.zeros(mask_pred_tensor.shape)
                mask_pred_numpy_bin[mask_pred_tensor>=0.5]=1

                save_image(mask_pred_tensor, seg_results_raw_path+n_img_name+'.png')
                save_image(mask_pred_numpy_bin, seg_results_raw_binary_path+n_img_name+'.png')

            pbar.update(imgs.shape[0])

def test_net(results_dir, batch_size, num_workers, device, dataset_train, dataset_test, image_size, job_name, threshold, checkpoint_mode, mask_or=True, train_or=False, crop_csv=''):

     


    data_path = results_dir+ "M2/binary_vessel/"
    test_dir = results_dir+ "M1/Good_quality/"
    
    mask_dir = "./data/{}/test/mask/".format(dataset_test)
    test_label = "./data/{}/test/1st_manual/".format(dataset_test)
    FD_list = []
    Name_list = []
    VD_list = []
    
    dataset_data = SEDataset_out(test_dir, test_label, mask_dir, image_size, dataset_test, threshold, uniform='True', crop_csv=crop_csv, train_or=False)
    test_loader = DataLoader(dataset_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, drop_last=False)
    
    dir_checkpoint_1= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,24)
    dir_checkpoint_2= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,26)
    dir_checkpoint_3= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,28)
    dir_checkpoint_4= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,30)
    dir_checkpoint_5= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,32)
    dir_checkpoint_6= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,34)
    dir_checkpoint_7= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,36)
    dir_checkpoint_8= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,38)
    dir_checkpoint_9= Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,40)
    dir_checkpoint_10=Path(__file__).parent / "./Saved_model/train_on_{}/{}_savebest_randomseed_{}/G_best_F1_epoch.pth".format(dataset_train,job_name,42)
    
    
    net_1 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_2 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_3 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_4 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_5 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_6 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_7 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_8 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_9 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_10 = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)

    net_1.load_state_dict(torch.load(dir_checkpoint_1, map_location = device))
    net_1.eval()
    net_1.to(device=device)
    net_2.load_state_dict(torch.load(dir_checkpoint_2, map_location = device))
    net_2.eval()
    net_2.to(device=device)
    net_3.load_state_dict(torch.load(dir_checkpoint_3, map_location = device))
    net_3.eval()
    net_3.to(device=device)
    net_4.load_state_dict(torch.load(dir_checkpoint_4, map_location = device))
    net_4.eval()
    net_4.to(device=device)
    net_5.load_state_dict(torch.load(dir_checkpoint_5, map_location = device))
    net_5.eval()
    net_5.to(device=device)
    net_6.load_state_dict(torch.load(dir_checkpoint_6, map_location = device))
    net_6.eval()
    net_6.to(device=device)
    net_7.load_state_dict(torch.load(dir_checkpoint_7, map_location = device))
    net_7.eval()
    net_7.to(device=device)
    net_8.load_state_dict(torch.load(dir_checkpoint_8, map_location = device))
    net_8.eval()
    net_8.to(device=device)
    net_9.load_state_dict(torch.load(dir_checkpoint_9, map_location = device))
    net_9.eval()
    net_9.to(device=device)
    net_10.load_state_dict(torch.load(dir_checkpoint_10, map_location = device))
    net_10.eval()
    net_10.to(device=device)
        
    segment_fundus(data_path, net_1, net_2, net_3, net_4, net_5, net_6, net_7, net_8, net_9, net_10, test_loader, device, dataset_train, job_name, mask_or, train_or)
    
    FD_list, Name_list, VD_list, width_cal = filter_frag(data_path)
    
    if not os.path.exists('{}M3/'.format(results_dir)):
        os.makedirs('{}M3/'.format(results_dir))
                            
    Data4stage2 = pd.DataFrame({'Image_id':Name_list, 'FD_boxC':FD_list, 'Vessel_Density':VD_list, 'Average_width':width_cal})
    Data4stage2.to_csv('{}M3/Binary_Features_Measurement.csv'.format(results_dir), index = None, encoding='utf8')
    
    #Optic_disk_loc(data_path)
        
class M2_VS_args():

    def __init__(self, cfg): 

        self.epochs = 1
        self.batchsize = cfg.batch_size
        self.lr = 2e-4
        self.load = False
        self.dis = "unet"
        self.jn = "20210630_uniform_thres40_ALL-SIX"
        self.worker = cfg.worker
        self.save = "best"
        self.ttmode = "trainandtest"
        self.pthreshold = 40.0
        self.dataset = "ALL-SIX"    
        self.seed = 42
        self.dataset_test = "ALL-SIX" 
        self.val = 10.0
        self.uniform = "True" 
        self.data_path = "{}M2/binary_vessel/".format(cfg.results_dir) 
        self.alpha = 0.08
        self.beta = 1.1
        self.gamma = 0.5       
        self.results = cfg.results_dir


def M2_vessel_seg(cfg):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = M2_VS_args(cfg)

    device = torch.device(cfg.device)
    logging.info(f'Using device {device}')

    image_size = Define_image_size(args.uniform, args.dataset)
    lr = args.lr

    test_csv= '{}M1/Good_quality/image_list.csv'.format(cfg.results_dir)

    net_G = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
    net_D = Discriminator(input_channels=4, n_filters = 32, n_classes=1, bilinear=False)


    if args.load:
        net_G.load_state_dict(
            torch.load(args.load, map_location=device)
        )

        net_D.load_state_dict(
            torch.load(args.load, map_location=device)
        )


    net_G.to(device=device)
    net_D.to(device=device)
    
    test_net(results_dir=cfg.results_dir,
             batch_size=args.batchsize,
             num_workers=args.worker,
             device=device,
             dataset_train=args.dataset,
             dataset_test=args.dataset_test, 
             image_size=image_size, 
             job_name=args.jn,
             threshold=args.pthreshold,
             checkpoint_mode=args.save,
             mask_or=True, 
             train_or=False,
             crop_csv=test_csv)
 