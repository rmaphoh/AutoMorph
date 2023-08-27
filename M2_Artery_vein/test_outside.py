'''
yukun 20220705
'''

import torch.nn.functional as F
import argparse
import logging
import shutil
import os
import cv2
import torchvision
import torch
import numpy as np
from tqdm import tqdm
from scripts.model import Generator_main, Generator_branch
from scripts.dataset import LearningAVSegData_OOD
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
from skimage import io
from scripts.utils import Define_image_size
from FD_cal import fractal_dimension,vessel_density
from skimage.morphology import skeletonize,remove_small_objects



def filter_frag(data_path):
    if os.path.isdir(data_path + 'raw/.ipynb_checkpoints'):
        shutil.rmtree(data_path + 'raw/.ipynb_checkpoints')

    image_list=os.listdir(data_path + 'raw')
    FD_cal_r=[]
    name_list=[]
    VD_cal_r=[]
    FD_cal_b=[]
    VD_cal_b=[]
    width_cal_r=[]
    width_cal_b=[]
    
    for i in sorted(image_list):
        img=io.imread(data_path + 'resized/' + i).astype(np.int64)
        img = cv2.resize(img,(912,912),interpolation = cv2.INTER_NEAREST)
        img2=img>0
        img_r = img2[...,0] + img2[...,1]
        img_b = img2[...,2] + img2[...,1]
        img_r = remove_small_objects(img_r, 30, connectivity=5)
        img_b = remove_small_objects(img_b, 30, connectivity=5)
        
        if not os.path.isdir(data_path + 'artery_binary_process/'):
            os.makedirs(data_path + 'artery_binary_process/') 
        io.imsave(data_path + 'artery_binary_process/' + i , 255*(img_r.astype('uint8')),check_contrast=False)
        if not os.path.isdir(data_path + 'vein_binary_process/'):
            os.makedirs(data_path + 'vein_binary_process/') 
        io.imsave(data_path + 'vein_binary_process/' + i , 255*(img_b.astype('uint8')),check_contrast=False)
        
        skeleton_r = skeletonize(img_r)
        skeleton_b = skeletonize(img_b)
        
        if not os.path.isdir(data_path + 'artery_binary_skeleton/'):
            os.makedirs(data_path + 'artery_binary_skeleton/') 
        io.imsave(data_path + 'artery_binary_skeleton/' + i, 255*(skeleton_r.astype('uint8')),check_contrast=False)
        if not os.path.isdir(data_path + 'vein_binary_skeleton/'):
            os.makedirs(data_path + 'vein_binary_skeleton/') 
        io.imsave(data_path + 'vein_binary_skeleton/' + i, 255*(skeleton_b.astype('uint8')),check_contrast=False)
        
        
        FD_boxcounting_r = fractal_dimension(img_r)
        FD_boxcounting_b = fractal_dimension(img_b)
        VD_r = vessel_density(img_r)
        VD_b = vessel_density(img_b)
        width_r = np.sum(img_r)/np.sum(skeleton_r)
        width_b = np.sum(img_b)/np.sum(skeleton_b)
        
        FD_cal_r.append(FD_boxcounting_r)
        name_list.append(i)
        VD_cal_r.append(VD_r)
        FD_cal_b.append(FD_boxcounting_b)
        VD_cal_b.append(VD_b)
        width_cal_r.append(width_r)
        width_cal_b.append(width_b)
    
    return FD_cal_r,name_list,VD_cal_r,FD_cal_b,VD_cal_b,width_cal_r,width_cal_b



def test_net(net_G_1, net_G_A_1, net_G_V_1, net_G_2, net_G_A_2, net_G_V_2, net_G_3, net_G_A_3, net_G_V_3, net_G_4, net_G_A_4, net_G_V_4, net_G_5, net_G_A_5, net_G_V_5, net_G_6, net_G_A_6, net_G_V_6, net_G_7, net_G_A_7, net_G_V_7, net_G_8, net_G_A_8, net_G_V_8, loader, device, mode, dataset):

    
    n_val = len(loader) 

    num = 0
    
    seg_results_small_path = '../Results/M2/artery_vein/resized/'
    seg_results_raw_path = '../Results/M2/artery_vein/raw/'
    
    if not os.path.isdir(seg_results_small_path):
        os.makedirs(seg_results_small_path)

    if not os.path.isdir(seg_results_raw_path):
        os.makedirs(seg_results_raw_path)

    seg_uncertainty_small_path = '../Results/M2/artery_vein/resize_uncertainty/'        
    if not os.path.isdir(seg_uncertainty_small_path):
        os.makedirs(seg_uncertainty_small_path)
    
    seg_uncertainty_raw_path = '../Results/M2/artery_vein/raw_uncertainty/'
    
    if not os.path.isdir(seg_uncertainty_raw_path):
        os.makedirs(seg_uncertainty_raw_path)
        
        
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            ori_width=batch['width']
            ori_height=batch['height']
            img_name = batch['name']
            mask_pred_tensor_small_all = 0
            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():

                num +=1
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_1(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_1(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_1(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_1 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_1.type(torch.FloatTensor)
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_2(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_2(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_2(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_2 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_2.type(torch.FloatTensor)
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_3(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_3(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_3(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_3 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_3.type(torch.FloatTensor)                
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_4(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_4(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_4(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_4 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_4.type(torch.FloatTensor)    
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_5(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_5(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_5(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_5 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_5.type(torch.FloatTensor)    
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_6(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_6(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_6(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_6 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_6.type(torch.FloatTensor)   
                
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_7(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_7(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_7(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_7 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_7.type(torch.FloatTensor)   
                
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_8(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_8(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_8(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_8 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_8.type(torch.FloatTensor)   
                
                mask_pred_tensor_small_all = (mask_pred_tensor_small_all/8).to(device=device)
                
                uncertainty_map = torch.sqrt((torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_1)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_2)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_3)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_4)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_5)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_6)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_7)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_8))/8)
            
                _,prediction_decode = torch.max(mask_pred_tensor_small_all, 1)
                prediction_decode=prediction_decode.type(torch.FloatTensor)
                
                
                
                if len(prediction_decode.size())==3:
                    torch.unsqueeze(prediction_decode,0)
                if len(uncertainty_map.size())==3:
                    torch.unsqueeze(uncertainty_map,0)
                    
                n_img = prediction_decode.shape[0]
                
                for i in range(n_img):
                    
                    save_image(uncertainty_map[i,...]*255, seg_uncertainty_small_path+img_name[i]+'.png')
                    save_image(uncertainty_map[i,1,...]*255, seg_uncertainty_small_path+img_name[i]+'_artery.png')
                    save_image(uncertainty_map[i,2,...]*255, seg_uncertainty_small_path+img_name[i]+'_vein.png')
                    
                    uncertainty_img = Image.open(seg_uncertainty_small_path+img_name[i]+'.png')
                    uncertainty_img = uncertainty_img.resize((int(ori_width[i]),int(ori_height[i])))
                    uncertainty_tensor = torchvision.transforms.ToTensor()(uncertainty_img)
                    save_image(uncertainty_tensor, seg_uncertainty_raw_path+img_name[i]+'.png')
                
                
                    img_r = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    img_g = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    img_b = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    
                    
                    img_r[prediction_decode[i,...]==1]=255
                    img_b[prediction_decode[i,...]==2]=255
                    img_g[prediction_decode[i,...]==3]=255

                    img_b = remove_small_objects(img_b>0, 30, connectivity=5)
                    img_r = remove_small_objects(img_r>0, 30, connectivity=5)

                    img_ = np.concatenate((img_b[...,np.newaxis], img_g[...,np.newaxis], img_r[...,np.newaxis]), axis=2)
                    
                    cv2.imwrite(seg_results_small_path+ img_name[i]+ '.png', np.float32(img_)*255)
                    
                    img_ww = cv2.resize(np.float32(img_)*255, (int(ori_width[i]),int(ori_height[i])), interpolation = cv2.INTER_NEAREST)
                    cv2.imwrite(seg_results_raw_path+ img_name[i]+ '.png', img_ww)
                
                
                
                pbar.update(imgs.shape[0])



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', type=int, default=6, help='Batch size', dest='batchsize')
    parser.add_argument('--job_name', type=str, default='J', help='type of discriminator', dest='jn')
    parser.add_argument('--dataset', type=str, help='test dataset name', dest='dataset')
    parser.add_argument('--checkstart', type=int, help='test dataset name', dest='CS')
    parser.add_argument('--uniform', type=str, default='False', help='whether to uniform the image size', dest='uniform')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Using device {device}')

    img_size = Define_image_size(args.uniform, args.dataset)
    dataset_name = args.dataset
    checkpoint_saved = dataset_name + '/' +args.jn + '/Discriminator_unet/'
    csv_save = 'test_csv/' + args.jn

    if not os.path.isdir(csv_save):
        os.makedirs(csv_save)

    test_dir= '../Results/M1/Good_quality/'
    test_label = "./data/{}/test/1st_manual/".format(dataset_name)
    test_mask =  "./data/{}/test/mask/".format(dataset_name)

    mode = 'whole'



    dataset = LearningAVSegData_OOD(test_dir, test_label, test_mask, img_size, dataset_name=dataset_name, train_or=False)
    test_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=16, pin_memory=False, drop_last=False)


    net_G_1 = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A_1 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V_1 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    net_G_2 = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A_2 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V_2 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    net_G_3 = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A_3 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V_3 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    net_G_4 = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A_4 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V_4 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    net_G_5 = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A_5 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V_5 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    net_G_6 = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A_6 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V_6 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    net_G_7 = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A_7 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V_7 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    net_G_8 = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A_8 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V_8 = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)


    checkpoint_saved_1="./ALL-AV/{}_{}/Discriminator_unet/".format( args.jn,28)
    checkpoint_saved_2="./ALL-AV/{}_{}/Discriminator_unet/".format( args.jn,30)
    checkpoint_saved_3="./ALL-AV/{}_{}/Discriminator_unet/".format( args.jn,32)
    checkpoint_saved_4="./ALL-AV/{}_{}/Discriminator_unet/".format( args.jn,34)
    checkpoint_saved_5="./ALL-AV/{}_{}/Discriminator_unet/".format( args.jn,36)
    checkpoint_saved_6="./ALL-AV/{}_{}/Discriminator_unet/".format( args.jn,38)
    checkpoint_saved_7="./ALL-AV/{}_{}/Discriminator_unet/".format( args.jn,40)
    checkpoint_saved_8="./ALL-AV/{}_{}/Discriminator_unet/".format( args.jn,42)


    for i in range(1):
        net_G_1.load_state_dict(torch.load(  checkpoint_saved_1 + 'CP_best_F1_all.pth'))
        net_G_A_1.load_state_dict(torch.load( checkpoint_saved_1 + 'CP_best_F1_A.pth'))
        net_G_V_1.load_state_dict(torch.load(checkpoint_saved_1 + 'CP_best_F1_V.pth'))
        net_G_1.eval()
        net_G_A_1.eval()
        net_G_V_1.eval()
        net_G_1.to(device=device)
        net_G_A_1.to(device=device)
        net_G_V_1.to(device=device)

        net_G_2.load_state_dict(torch.load(  checkpoint_saved_2 + 'CP_best_F1_all.pth'))
        net_G_A_2.load_state_dict(torch.load( checkpoint_saved_2 + 'CP_best_F1_A.pth'))
        net_G_V_2.load_state_dict(torch.load(checkpoint_saved_2 + 'CP_best_F1_V.pth'))
        net_G_2.eval()
        net_G_A_2.eval()
        net_G_V_2.eval()
        net_G_2.to(device=device)
        net_G_A_2.to(device=device)
        net_G_V_2.to(device=device)
        
        net_G_3.load_state_dict(torch.load(  checkpoint_saved_3 + 'CP_best_F1_all.pth'))
        net_G_A_3.load_state_dict(torch.load( checkpoint_saved_3 + 'CP_best_F1_A.pth'))
        net_G_V_3.load_state_dict(torch.load(checkpoint_saved_3 + 'CP_best_F1_V.pth'))
        net_G_3.eval()
        net_G_A_3.eval()
        net_G_V_3.eval()
        net_G_3.to(device=device)
        net_G_A_3.to(device=device)
        net_G_V_3.to(device=device)
        
        net_G_4.load_state_dict(torch.load(  checkpoint_saved_4 + 'CP_best_F1_all.pth'))
        net_G_A_4.load_state_dict(torch.load( checkpoint_saved_4 + 'CP_best_F1_A.pth'))
        net_G_V_4.load_state_dict(torch.load(checkpoint_saved_4 + 'CP_best_F1_V.pth'))
        net_G_4.eval()
        net_G_A_4.eval()
        net_G_V_4.eval()
        net_G_4.to(device=device)
        net_G_A_4.to(device=device)
        net_G_V_4.to(device=device)
        
        net_G_5.load_state_dict(torch.load(  checkpoint_saved_5 + 'CP_best_F1_all.pth'))
        net_G_A_5.load_state_dict(torch.load( checkpoint_saved_5 + 'CP_best_F1_A.pth'))
        net_G_V_5.load_state_dict(torch.load(checkpoint_saved_5 + 'CP_best_F1_V.pth'))
        net_G_5.eval()
        net_G_A_5.eval()
        net_G_V_5.eval()
        net_G_5.to(device=device)
        net_G_A_5.to(device=device)
        net_G_V_5.to(device=device)
        
        net_G_6.load_state_dict(torch.load(  checkpoint_saved_6 + 'CP_best_F1_all.pth'))
        net_G_A_6.load_state_dict(torch.load( checkpoint_saved_6 + 'CP_best_F1_A.pth'))
        net_G_V_6.load_state_dict(torch.load(checkpoint_saved_6 + 'CP_best_F1_V.pth'))
        net_G_6.eval()
        net_G_A_6.eval()
        net_G_V_6.eval()
        net_G_6.to(device=device)
        net_G_A_6.to(device=device)
        net_G_V_6.to(device=device)
        
        net_G_7.load_state_dict(torch.load(  checkpoint_saved_7 + 'CP_best_F1_all.pth'))
        net_G_A_7.load_state_dict(torch.load( checkpoint_saved_7 + 'CP_best_F1_A.pth'))
        net_G_V_7.load_state_dict(torch.load(checkpoint_saved_7 + 'CP_best_F1_V.pth'))
        net_G_7.eval()
        net_G_A_7.eval()
        net_G_V_7.eval()
        net_G_7.to(device=device)
        net_G_A_7.to(device=device)
        net_G_V_7.to(device=device)
        
        net_G_8.load_state_dict(torch.load(  checkpoint_saved_8 + 'CP_best_F1_all.pth'))
        net_G_A_8.load_state_dict(torch.load( checkpoint_saved_8 + 'CP_best_F1_A.pth'))
        net_G_V_8.load_state_dict(torch.load(checkpoint_saved_8 + 'CP_best_F1_V.pth'))
        net_G_8.eval()
        net_G_A_8.eval()
        net_G_V_8.eval()
        net_G_8.to(device=device)
        net_G_A_8.to(device=device)
        net_G_V_8.to(device=device)
        
        if mode != 'vessel':
            test_net(net_G_1, net_G_A_1, net_G_V_1, net_G_2, net_G_A_2, net_G_V_2, net_G_3, net_G_A_3, net_G_V_3, net_G_4, net_G_A_4, net_G_V_4, net_G_5, net_G_A_5, net_G_V_5, net_G_6, net_G_A_6, net_G_V_6, net_G_7, net_G_A_7, net_G_V_7, net_G_8, net_G_A_8, net_G_V_8, loader=test_loader, device=device, mode=mode,dataset=dataset_name)


        FD_list_r,name_list,VD_list_r,FD_list_v,VD_list_b,width_cal_r,width_cal_b = filter_frag(data_path='../Results/M2/artery_vein/')
        
        
        #Data4stage2 = pd.DataFrame({'Image_id':name_list, 'FD_boxC_artery':FD_list_r, 'Vessel_Density_artery':VD_list_r})
        #Data4stage2.to_csv('../Results/M3/Artery_Features_Measurement.csv', index = None, encoding='utf8')
        
        #Data4stage2 = pd.DataFrame({'Image_id':name_list, 'FD_boxC_vein':FD_list_v, 'Vessel_Density_vein':VD_list_b})
        #Data4stage2.to_csv('../Results/M3/Vein_Features_Measurement.csv', index = None, encoding='utf8')

