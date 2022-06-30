import os, json, sys
import os.path as osp
import argparse
import warnings
from tqdm import tqdm
import cv2
import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.color import label2rgb
import shutil
from PIL import Image
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torchvision
from models.get_model import get_arch
from utils.get_loaders import get_test_dataset
from utils.model_saving_loading import load_model
from skimage import filters, measure
import skimage
import pandas as pd
from skimage.morphology import skeletonize,remove_small_objects
# argument parsing
parser = argparse.ArgumentParser()
required_named = parser.add_argument_group('required arguments')
parser.add_argument('--experiment_path', help='experiments/subfolder where checkpoint is', default=None)
parser.add_argument('--config_file', type=str, default=None,
                    help='experiments/name_of_config_file, overrides everything')
# in case no config file is passed
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--results_path', type=str, default='results', help='path to save predictions (defaults to results')


def intersection(mask,vessel_, it_x, it_y):
    """
    Remove the intersection in case the whole vessel is too long
    """
    x_less = max(0, it_x - 1)
    y_less = max(0, it_y - 1)
    x_more = min(vessel_.shape[0] - 1, it_x + 1)
    y_more = min(vessel_.shape[1] - 1, it_y + 1)

    active_neighbours = (vessel_[x_less, y_less]>0).astype('float')+ \
                        (vessel_[x_less, it_y]>0).astype('float')+ \
                        (vessel_[x_less, y_more]>0).astype('float')+ \
                        (vessel_[it_x, y_less]>0).astype('float')+ \
                        (vessel_[it_x, y_more]>0).astype('float')+ \
                        (vessel_[x_more, y_less]>0).astype('float')+ \
                        (vessel_[x_more, it_y]>0).astype('float')+ \
                        (vessel_[x_more, y_more]>0).astype('float')

    if active_neighbours > 2:
        cv2.circle(mask,(it_y,it_x),radius=1,color=(0,0,0),thickness=-1)
    

    return mask,active_neighbours   
    
    

def optic_disc_centre(result_path, binary_vessel_path, artery_vein_path):
    if os.path.exists(result_path+'.ipynb_checkpoints'):
        shutil.rmtree(result_path+'.ipynb_checkpoints')
        
    optic_binary_result_path = '../Results/M3/Disc_centred/'
    macular_binary_result_path = '../Results/M3/Macular_centred/'
    
    B_optic_process_binary_vessel_path = binary_vessel_path + 'Zone_B_disc_centred_binary_process/'
    B_optic_process_artery_path = artery_vein_path + 'Zone_B_disc_centred_artery_process/'
    B_optic_process_vein_path = artery_vein_path + 'Zone_B_disc_centred_vein_process/'
    
    B_optic_skeleton_binary_vessel_path = binary_vessel_path + 'Zone_B_disc_centred_binary_skeleton/'
    B_optic_skeleton_artery_path = artery_vein_path + 'Zone_B_disc_centred_artery_skeleton/'
    B_optic_skeleton_vein_path = artery_vein_path + 'Zone_B_disc_centred_vein_skeleton/'

    C_optic_process_binary_vessel_path = binary_vessel_path + 'Zone_C_disc_centred_binary_process/'
    C_optic_process_artery_path = artery_vein_path + 'Zone_C_disc_centred_artery_process/'
    C_optic_process_vein_path = artery_vein_path + 'Zone_C_disc_centred_vein_process/'
    
    C_optic_skeleton_binary_vessel_path = binary_vessel_path + 'Zone_C_disc_centred_binary_skeleton/'
    C_optic_skeleton_artery_path = artery_vein_path + 'Zone_C_disc_centred_artery_skeleton/'
    C_optic_skeleton_vein_path = artery_vein_path + 'Zone_C_disc_centred_vein_skeleton/'
    
    
    macular_process_binary_vessel_path = binary_vessel_path + 'macular_centred_binary_process/'
    macular_process_artery_path = artery_vein_path + 'macular_centred_artery_process/'
    macular_process_vein_path = artery_vein_path + 'macular_centred_vein_process/'
    
    macular_skeleton_binary_vessel_path = binary_vessel_path + 'macular_centred_binary_skeleton/'
    macular_skeleton_artery_path = artery_vein_path + 'macular_centred_artery_skeleton/'
    macular_skeleton_vein_path = artery_vein_path + 'macular_centred_vein_skeleton/'
    
    #2021/11/2
    zone_b_macular_process_binary_vessel_path = binary_vessel_path + 'macular_Zone_B_centred_binary_process/'
    zone_b_macular_process_artery_path = artery_vein_path + 'macular_Zone_B_centred_artery_process/'
    zone_b_macular_process_vein_path = artery_vein_path + 'macular_Zone_B_centred_vein_process/'
    
    zone_b_macular_skeleton_binary_vessel_path = binary_vessel_path + 'macular_Zone_B_centred_binary_skeleton/'
    zone_b_macular_skeleton_artery_path = artery_vein_path + 'macular_Zone_B_centred_artery_skeleton/'
    zone_b_macular_skeleton_vein_path = artery_vein_path + 'macular_Zone_B_centred_vein_skeleton/'
    #2021/11/2
    zone_c_macular_process_binary_vessel_path = binary_vessel_path + 'macular_Zone_C_centred_binary_process/'
    zone_c_macular_process_artery_path = artery_vein_path + 'macular_Zone_C_centred_artery_process/'
    zone_c_macular_process_vein_path = artery_vein_path + 'macular_Zone_C_centred_vein_process/'
    
    zone_c_macular_skeleton_binary_vessel_path = binary_vessel_path + 'macular_Zone_C_centred_binary_skeleton/'
    zone_c_macular_skeleton_artery_path = artery_vein_path + 'macular_Zone_C_centred_artery_skeleton/'
    zone_c_macular_skeleton_vein_path = artery_vein_path + 'macular_Zone_C_centred_vein_skeleton/'
    
    if not os.path.exists(optic_binary_result_path):
        os.makedirs(optic_binary_result_path)
    if not os.path.exists(macular_binary_result_path):
        os.makedirs(macular_binary_result_path)
    
    if not os.path.exists(B_optic_process_binary_vessel_path):
        os.makedirs(B_optic_process_binary_vessel_path)
    if not os.path.exists(B_optic_process_artery_path):
        os.makedirs(B_optic_process_artery_path)
    if not os.path.exists(B_optic_process_vein_path):
        os.makedirs(B_optic_process_vein_path)
    if not os.path.exists(B_optic_skeleton_binary_vessel_path):
        os.makedirs(B_optic_skeleton_binary_vessel_path)
    if not os.path.exists(B_optic_skeleton_artery_path):
        os.makedirs(B_optic_skeleton_artery_path)
    if not os.path.exists(B_optic_skeleton_vein_path):
        os.makedirs(B_optic_skeleton_vein_path)

    if not os.path.exists(C_optic_process_binary_vessel_path):
        os.makedirs(C_optic_process_binary_vessel_path)
    if not os.path.exists(C_optic_process_artery_path):
        os.makedirs(C_optic_process_artery_path)
    if not os.path.exists(C_optic_process_vein_path):
        os.makedirs(C_optic_process_vein_path)
    if not os.path.exists(C_optic_skeleton_binary_vessel_path):
        os.makedirs(C_optic_skeleton_binary_vessel_path)
    if not os.path.exists(C_optic_skeleton_artery_path):
        os.makedirs(C_optic_skeleton_artery_path)
    if not os.path.exists(C_optic_skeleton_vein_path):
        os.makedirs(C_optic_skeleton_vein_path)
        
        
    if not os.path.exists(macular_process_binary_vessel_path):
        os.makedirs(macular_process_binary_vessel_path)
    if not os.path.exists(macular_process_artery_path):
        os.makedirs(macular_process_artery_path)
    if not os.path.exists(macular_process_vein_path):
        os.makedirs(macular_process_vein_path)
    if not os.path.exists(macular_skeleton_binary_vessel_path):
        os.makedirs(macular_skeleton_binary_vessel_path)
    if not os.path.exists(macular_skeleton_artery_path):
        os.makedirs(macular_skeleton_artery_path)
    if not os.path.exists(macular_skeleton_vein_path):
        os.makedirs(macular_skeleton_vein_path)
    

    if not os.path.exists(zone_b_macular_process_binary_vessel_path):
        os.makedirs(zone_b_macular_process_binary_vessel_path)
    if not os.path.exists(zone_b_macular_process_artery_path):
        os.makedirs(zone_b_macular_process_artery_path)
    if not os.path.exists(zone_b_macular_process_vein_path):
        os.makedirs(zone_b_macular_process_vein_path)
    if not os.path.exists(zone_b_macular_skeleton_binary_vessel_path):
        os.makedirs(zone_b_macular_skeleton_binary_vessel_path)
    if not os.path.exists(zone_b_macular_skeleton_artery_path):
        os.makedirs(zone_b_macular_skeleton_artery_path)
    if not os.path.exists(zone_b_macular_skeleton_vein_path):
        os.makedirs(zone_b_macular_skeleton_vein_path)
        
    if not os.path.exists(zone_c_macular_process_binary_vessel_path):
        os.makedirs(zone_c_macular_process_binary_vessel_path)
    if not os.path.exists(zone_c_macular_process_artery_path):
        os.makedirs(zone_c_macular_process_artery_path)
    if not os.path.exists(zone_c_macular_process_vein_path):
        os.makedirs(zone_c_macular_process_vein_path)
    if not os.path.exists(zone_c_macular_skeleton_binary_vessel_path):
        os.makedirs(zone_c_macular_skeleton_binary_vessel_path)
    if not os.path.exists(zone_c_macular_skeleton_artery_path):
        os.makedirs(zone_c_macular_skeleton_artery_path)
    if not os.path.exists(zone_c_macular_skeleton_vein_path):
        os.makedirs(zone_c_macular_skeleton_vein_path)
        
    optic_vertical_CDR,optic_vertical_disc,optic_vertical_cup = [],[],[]
    optic_horizontal_CDR,optic_horizontal_disc,optic_horizontal_cup = [],[],[]
    
    macular_vertical_CDR,macular_vertical_disc,macular_vertical_cup = [],[],[]
    macular_horizontal_CDR,macular_horizontal_disc,macular_horizontal_cup = [],[],[]
    
    optic_centre_list = []
    macular_centre_list = []
    
    disc_cup_list = sorted(os.listdir(result_path))
    
    for i in disc_cup_list:
        path_ = result_path+i
        disc_cup_ = cv2.imread(path_)
        disc_cup_912 = cv2.resize(disc_cup_,(912,912),interpolation = cv2.INTER_NEAREST)
        
        #image_ = cv2.imread('../Results/M1/Good_quality/'+i)
        #IMAGE_912 = cv2.resize(image_,(912,912),interpolation = cv2.INTER_AREA)
        #disc_cup_912 = disc_cup_
        try:
    
            disc_ = disc_cup_912[...,2]
            cup_ = disc_cup_912[...,0]
            
            ## judgement the optic disc/cup segmentation
            disc_mask = measure.label(disc_)                       
            regions = measure.regionprops(disc_mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            if len(regions) > 1:
                for rg in regions[2:]:
                    disc_mask[rg.coords[:,0], rg.coords[:,1]] = 0
            disc_[disc_mask!=0] = 255
            
            cup_mask = measure.label(cup_)                       
            regions = measure.regionprops(cup_mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            if len(regions) > 1:
                for rg in regions[2:]:
                    cup_mask[rg.coords[:,0], rg.coords[:,1]] = 0
            cup_[cup_mask!=0] = 255
            
            disc_index = np.where(disc_>0)
            disc_index_width = disc_index[1]
            disc_index_height = disc_index[0]
            disc_horizontal_width = np.max(disc_index_width)-np.min(disc_index_width)
            disc_vertical_height = np.max(disc_index_height)-np.min(disc_index_height)

            
            cup_index = np.where(cup_>0)
            cup_index_width = cup_index[1]
            cup_index_height = cup_index[0]
            cup_horizontal_width = np.max(cup_index_width)-np.min(cup_index_width)
            cup_vertical_height = np.max(cup_index_height)-np.min(cup_index_height)
            
            cup_width_centre = np.mean(cup_index_width)
            cup_height_centre = np.mean(cup_index_height)
            
            #print('@@@@@@@@@@@@@@@@@@@@@@@@')
            #print(i)
            #print(disc_horizontal_width<(disc_.shape[0]/3))
            #print(disc_vertical_height<(disc_.shape[1]/3))
            #print(cup_width_centre<=np.max(disc_index_width))
            #print(cup_width_centre>=np.min(disc_index_width))
            #print(cup_height_centre<=np.max(disc_index_height))
            #print(cup_height_centre>=np.min(disc_index_height))
                
            if disc_horizontal_width<(disc_.shape[0]/3) and disc_vertical_height<(disc_.shape[1]/3) and cup_width_centre<=np.max(disc_index_width) and cup_width_centre>=np.min(disc_index_width) and cup_height_centre<=np.max(disc_index_height) and cup_height_centre>=np.min(disc_index_height) and cup_vertical_height<disc_vertical_height and cup_horizontal_width<disc_horizontal_width:
            

                whole_index = np.where(disc_cup_912>0)
                whole_index_width = whole_index[1]
                whole_index_height = whole_index[0]

                horizontal_distance = np.absolute(np.mean(whole_index_height)-disc_cup_912.shape[1]/2)
                vertical_distance = np.absolute(np.mean(whole_index_width)-disc_cup_912.shape[0]/2)
                distance_ = np.sqrt(np.square(horizontal_distance)+np.square(vertical_distance))


                binary_process_ = cv2.imread(binary_vessel_path+'binary_process/'+i)[...,0]
                artery_process_ = cv2.imread(artery_vein_path+'artery_binary_process/'+i)[...,0]
                vein_process_ = cv2.imread(artery_vein_path+'vein_binary_process/'+i)[...,0]

                binary_skeleton_ = cv2.imread(binary_vessel_path+'binary_skeleton/'+i)[...,0]
                artery_skeleton_ = cv2.imread(artery_vein_path+'artery_binary_skeleton/'+i)[...,0]
                vein_skeleton_ = cv2.imread(artery_vein_path+'vein_binary_skeleton/'+i)[...,0]

                # remove the intersection of binary_skeleton_
                ignored_pixels = 1

                mask_ = np.ones((binary_skeleton_.shape))
                for it_x in range(ignored_pixels, mask_.shape[0] - ignored_pixels):
                    for it_y in range(ignored_pixels, mask_.shape[1] - ignored_pixels):
                        if binary_skeleton_[it_x, it_y] > 0:
                            mask,active_neighbours = intersection(mask_,binary_skeleton_, it_x, it_y)

                binary_skeleton_ = binary_skeleton_ * mask

                # remove the intersection of artery_skeleton_
                mask_ = np.ones((artery_skeleton_.shape))
                for it_x in range(ignored_pixels, mask_.shape[0] - ignored_pixels):
                    for it_y in range(ignored_pixels, mask_.shape[1] - ignored_pixels):
                        if artery_skeleton_[it_x, it_y] > 0:
                            mask,active_neighbours = intersection(mask_,artery_skeleton_, it_x, it_y)

                artery_skeleton_ = artery_skeleton_ * mask

                # remove the intersection of vein_skeleton_
                mask_ = np.ones((vein_skeleton_.shape))
                for it_x in range(ignored_pixels, mask_.shape[0] - ignored_pixels):
                    for it_y in range(ignored_pixels, mask_.shape[1] - ignored_pixels):
                        if vein_skeleton_[it_x, it_y] > 0:
                            mask,active_neighbours = intersection(mask_,vein_skeleton_, it_x, it_y)

                vein_skeleton_ = vein_skeleton_ * mask


                zone_mask_B = np.zeros(binary_process_.shape)
                zone_mask_C = np.zeros(binary_process_.shape)
                zone_centre = (int(np.mean(whole_index_width)), int(np.mean(whole_index_height)))
                radius = max(int(disc_horizontal_width/2),int(disc_vertical_height/2))
                cv2.circle(zone_mask_B,zone_centre,radius=3*radius,color=(255,255,255),thickness=-1)
                cv2.circle(zone_mask_B,zone_centre,radius=2*radius,color=(0,0,0),thickness=-1)
                zone_mask_B = zone_mask_B/255

                #binary_process_B = binary_process_*zone_mask_B + IMAGE_912[...,0]*0.5
                #artery_process_B = artery_process_*zone_mask_B + IMAGE_912[...,0]*0.5
                binary_process_B = binary_process_*zone_mask_B 
                artery_process_B = artery_process_*zone_mask_B 
                vein_process_B = vein_process_*zone_mask_B
                binary_skeleton_B = binary_skeleton_*zone_mask_B
                artery_skeleton_B = artery_skeleton_*zone_mask_B
                vein_skeleton_B = vein_skeleton_*zone_mask_B

                cv2.circle(zone_mask_C,zone_centre,radius=5*radius,color=(255,255,255),thickness=-1)
                cv2.circle(zone_mask_C,zone_centre,radius=2*radius,color=(0,0,0),thickness=-1)
                zone_mask_C = zone_mask_C/255            

                #binary_process_C = binary_process_*zone_mask_C + IMAGE_912[...,0]*0.5
                #artery_process_C = artery_process_*zone_mask_C + IMAGE_912[...,0]*0.5
                binary_process_C = binary_process_*zone_mask_C 
                artery_process_C = artery_process_*zone_mask_C 
                vein_process_C = vein_process_*zone_mask_C
                binary_skeleton_C = binary_skeleton_*zone_mask_C
                artery_skeleton_C = artery_skeleton_*zone_mask_C
                vein_skeleton_C = vein_skeleton_*zone_mask_C



                if (distance_/disc_cup_912.shape[1])<0.1:
                    optic_centre_list.append(i)
                    cv2.imwrite(B_optic_process_binary_vessel_path+i,binary_process_B)
                    cv2.imwrite(B_optic_process_artery_path+i,artery_process_B)
                    cv2.imwrite(B_optic_process_vein_path+i,vein_process_B)
                    cv2.imwrite(B_optic_skeleton_binary_vessel_path+i,binary_skeleton_B)
                    cv2.imwrite(B_optic_skeleton_artery_path+i,artery_skeleton_B)
                    cv2.imwrite(B_optic_skeleton_vein_path+i,vein_skeleton_B)

                    cv2.imwrite(C_optic_process_binary_vessel_path+i,binary_process_C)
                    cv2.imwrite(C_optic_process_artery_path+i,artery_process_C)
                    cv2.imwrite(C_optic_process_vein_path+i,vein_process_C)
                    cv2.imwrite(C_optic_skeleton_binary_vessel_path+i,binary_skeleton_C)
                    cv2.imwrite(C_optic_skeleton_artery_path+i,artery_skeleton_C)
                    cv2.imwrite(C_optic_skeleton_vein_path+i,vein_skeleton_C)            


                    optic_vertical_disc.append(disc_vertical_height)
                    optic_horizontal_disc.append(disc_horizontal_width)

                    optic_vertical_cup.append(cup_vertical_height)
                    optic_horizontal_cup.append(cup_horizontal_width)

                    optic_vertical_CDR.append(cup_vertical_height/disc_vertical_height)
                    optic_horizontal_CDR.append(cup_horizontal_width/disc_horizontal_width)
                #print('@@@@@@')
                #print(disc_cup_.shape[1]/2)
                #print(disc_cup_.shape[0]/2)
                #print(np.mean(whole_index_height))
                #print(np.mean(whole_index_width))
                #print(distance_)
                #print((distance_/disc_cup_912.shape[1]))


                else:
                    macular_centre_list.append(i)
                    cv2.imwrite(zone_b_macular_process_binary_vessel_path+i,binary_process_B)
                    cv2.imwrite(zone_b_macular_process_artery_path+i,artery_process_B)
                    cv2.imwrite(zone_b_macular_process_vein_path+i,vein_process_B)
                    cv2.imwrite(zone_b_macular_skeleton_binary_vessel_path+i,binary_skeleton_B)
                    cv2.imwrite(zone_b_macular_skeleton_artery_path+i,artery_skeleton_B)
                    cv2.imwrite(zone_b_macular_skeleton_vein_path+i,vein_skeleton_B)

                    cv2.imwrite(zone_c_macular_process_binary_vessel_path+i,binary_process_C)
                    cv2.imwrite(zone_c_macular_process_artery_path+i,artery_process_C)
                    cv2.imwrite(zone_c_macular_process_vein_path+i,vein_process_C)
                    cv2.imwrite(zone_c_macular_skeleton_binary_vessel_path+i,binary_skeleton_C)
                    cv2.imwrite(zone_c_macular_skeleton_artery_path+i,artery_skeleton_C)
                    cv2.imwrite(zone_c_macular_skeleton_vein_path+i,vein_skeleton_C)


                    shutil.copy(binary_vessel_path+'binary_process/'+i,macular_process_binary_vessel_path+i)
                    shutil.copy(artery_vein_path+'artery_binary_process/'+i,macular_process_artery_path+i)
                    shutil.copy(artery_vein_path+'vein_binary_process/'+i,macular_process_vein_path+i)
                    shutil.copy(binary_vessel_path+'binary_skeleton/'+i,macular_skeleton_binary_vessel_path+i)
                    shutil.copy(artery_vein_path+'artery_binary_skeleton/'+i,macular_skeleton_artery_path+i)
                    shutil.copy(artery_vein_path+'vein_binary_skeleton/'+i,macular_skeleton_vein_path+i)

                    macular_vertical_disc.append(disc_vertical_height)
                    macular_horizontal_disc.append(disc_horizontal_width)

                    macular_vertical_cup.append(cup_vertical_height)
                    macular_horizontal_cup.append(cup_horizontal_width)

                    macular_vertical_CDR.append(cup_vertical_height/disc_vertical_height)
                    macular_horizontal_CDR.append(cup_horizontal_width/disc_horizontal_width)
        
            
            else:
                macular_centre_list.append(i)
                shutil.copy(binary_vessel_path+'binary_process/'+i,macular_process_binary_vessel_path+i)
                shutil.copy(artery_vein_path+'artery_binary_process/'+i,macular_process_artery_path+i)
                shutil.copy(artery_vein_path+'vein_binary_process/'+i,macular_process_vein_path+i)
                shutil.copy(binary_vessel_path+'binary_skeleton/'+i,macular_skeleton_binary_vessel_path+i)
                shutil.copy(artery_vein_path+'artery_binary_skeleton/'+i,macular_skeleton_artery_path+i)
                shutil.copy(artery_vein_path+'vein_binary_skeleton/'+i,macular_skeleton_vein_path+i)

                macular_vertical_disc.append(-1)
                macular_horizontal_disc.append(-1)

                macular_vertical_cup.append(-1)
                macular_horizontal_cup.append(-1)

                macular_vertical_CDR.append(-1)
                macular_horizontal_CDR.append(-1)
                
        except:
            macular_centre_list.append(i)
            shutil.copy(binary_vessel_path+'binary_process/'+i,macular_process_binary_vessel_path+i)
            shutil.copy(artery_vein_path+'artery_binary_process/'+i,macular_process_artery_path+i)
            shutil.copy(artery_vein_path+'vein_binary_process/'+i,macular_process_vein_path+i)
            shutil.copy(binary_vessel_path+'binary_skeleton/'+i,macular_skeleton_binary_vessel_path+i)
            shutil.copy(artery_vein_path+'artery_binary_skeleton/'+i,macular_skeleton_artery_path+i)
            shutil.copy(artery_vein_path+'vein_binary_skeleton/'+i,macular_skeleton_vein_path+i)

            macular_vertical_disc.append(-1)
            macular_horizontal_disc.append(-1)

            macular_vertical_cup.append(-1)
            macular_horizontal_cup.append(-1)

            macular_vertical_CDR.append(-1)
            macular_horizontal_CDR.append(-1)
                
            
    Pd_optic_centre = pd.DataFrame({'Name':optic_centre_list, 'Disc_height':optic_vertical_disc, 'Disc_width':optic_horizontal_disc, 'Cup_height': optic_vertical_cup, 'Cup_width': optic_horizontal_cup, 'CDR_vertical': optic_vertical_CDR, 'CDR_horizontal': optic_horizontal_CDR})

    Pd_optic_centre.to_csv(optic_binary_result_path + 'Disc_cup_results.csv', index = None, encoding='utf8')
    
    
    Pd_macular_centre = pd.DataFrame({'Name':macular_centre_list, 'Disc_height':macular_vertical_disc, 'Disc_width':macular_horizontal_disc, 'Cup_height': macular_vertical_cup, 'Cup_width': macular_horizontal_cup, 'CDR_vertical': macular_vertical_CDR, 'CDR_horizontal': macular_horizontal_CDR})

    Pd_macular_centre.to_csv(macular_binary_result_path + 'Disc_cup_results.csv', index = None, encoding='utf8')        
    

    
def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm=confusion_matrix(true_vessel_arr, pred_vessel_arr)
    mse = mean_squared_error(true_vessel_arr, pred_vessel_arr)

    try:
        acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
        sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
        specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
        precision=1.*cm[1,1]/(cm[1,1]+cm[0,1])
        G = np.sqrt(sensitivity*specificity)
        F1_score = 2*precision*sensitivity/(precision+sensitivity)
        iou = 1.*cm[1,1]/(cm[1,0]+cm[0,1]+cm[1,1])
        return acc, sensitivity, specificity, precision, G, F1_score, mse, iou
    except:
        return 0,0,0,0,0,0,0,0
    
    
def evaluate_disc(results_path, label_path):
    if os.path.exists(results_path+'.ipynb_checkpoints'):
        shutil.rmtree(results_path+'.ipynb_checkpoints')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    seg_list = os.listdir(results_path)

    tot=[]
    sent=[]
    spet=[]
    pret=[]
    G_t=[]
    F1t=[]
    mset=[]
    iout=[]
    
    n_val = len(seg_list)
    
    for i in seg_list:
        label_name = i.split('.')[0] + '_OD.png'
        label_ = cv2.imread(label_path+label_name)/255
        label_=label_[...,0]
        seg_ = cv2.imread(results_path + i)
        seg_ = (seg_<255).astype('float')[...,0]
        
        print(np.unique(label_))
        print(np.unique(seg_))
        acc, sensitivity, specificity, precision, G, F1_score, mse, iou = misc_measures(label_.flatten(), seg_.flatten())
        
        tot.append(acc) 
        sent.append(sensitivity)
        spet.append(specificity)
        pret.append(precision)
        G_t.append(G)
        F1t.append(F1_score)
        mset.append(mse)
        iout.append(iou)
        
    
    Data4stage2 = pd.DataFrame({'ACC':tot, 'Sensitivity':sent, 'Specificity':spet, 'Precision': pret, 'G_value': G_t, \
                                    'F1-score': F1t, 'MSE': mset, 'IOU': iout})
    Data4stage2.to_csv('./results/IDRID_optic/performance.csv', index = None, encoding='utf8')
        
        
    #return tot / n_val, sent / n_val, spet / n_val, pret / n_val, G_t / n_val, F1t / n_val, auc_roct / n_val, auc_prt / n_val, iout/n_val, mset/n_val
        
def prediction_eval(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8, test_loader):
    
    n_val = len(test_loader)
    
    seg_results_small_path = '../Results/M2/optic_disc_cup/resized/'
    seg_results_raw_path = '../Results/M2/optic_disc_cup/raw/'
    
    if not os.path.isdir(seg_results_small_path):
        os.makedirs(seg_results_small_path)

    if not os.path.isdir(seg_results_raw_path):
        os.makedirs(seg_results_raw_path)

    seg_uncertainty_small_path = '../Results/M2/optic_disc_cup/resize_uncertainty/'        
    if not os.path.isdir(seg_uncertainty_small_path):
        os.makedirs(seg_uncertainty_small_path)
    
    seg_uncertainty_raw_path = '../Results/M2/optic_disc_cup/raw_uncertainty/'
    
    if not os.path.isdir(seg_uncertainty_raw_path):
        os.makedirs(seg_uncertainty_raw_path)
        
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            imgs = batch['image']
            img_name = batch['name']
            ori_width=batch['original_sz'][0]
            ori_height=batch['original_sz'][1]
            mask_pred_tensor_small_all = 0
            
            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():

                _,mask_pred = model_1(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_1 = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_1.type(torch.FloatTensor)
                
                
                _,mask_pred= model_2(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_2 = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_2.type(torch.FloatTensor)
                

                _,mask_pred = model_3(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_3 = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_3.type(torch.FloatTensor)                
                

                _,mask_pred = model_4(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_4 = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_4.type(torch.FloatTensor)    
                

                _,mask_pred = model_5(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_5 = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_5.type(torch.FloatTensor)    
                

                _,mask_pred = model_6(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_6 = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_6.type(torch.FloatTensor)   
                

                _,mask_pred = model_7(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_7 = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_7.type(torch.FloatTensor)   
                

                _,mask_pred = model_8(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_8 = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_8.type(torch.FloatTensor)   
                
                mask_pred_tensor_small_all = (mask_pred_tensor_small_all/8).to(device=device)
                
                #print(mask_pred_tensor_small_all.is_cuda)
                #print(mask_pred_tensor_small_1.is_cuda)
                
                uncertainty_map = torch.sqrt((torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_1)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_2)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_3)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_4)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_5)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_6)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_7)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_8))/8)
            
                _,prediction_decode = torch.max(mask_pred_tensor_small_all, 1)
                prediction_decode=prediction_decode.type(torch.FloatTensor)
                
                n_img = prediction_decode.shape[0]
                
                if len(prediction_decode.size())==3:
                    torch.unsqueeze(prediction_decode,0)
                    
                for i in range(n_img):
                    
                    save_image(uncertainty_map[i,...]*255, seg_uncertainty_small_path+img_name[i]+'.png')
                    save_image(uncertainty_map[i,1,...]*255, seg_uncertainty_small_path+img_name[i]+'_disc.png')
                    save_image(uncertainty_map[i,2,...]*255, seg_uncertainty_small_path+img_name[i]+'_cup.png')
                    
                    uncertainty_img = Image.open(seg_uncertainty_small_path+img_name[i]+'.png')
                    uncertainty_img = uncertainty_img.resize((int(ori_width[i]),int(ori_height[i])))
                    uncertainty_tensor = torchvision.transforms.ToTensor()(uncertainty_img)
                    save_image(uncertainty_tensor, seg_uncertainty_raw_path+img_name[i]+'.png')
                    
                    img_r = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    img_g = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    img_b = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    
                    
                    img_r[prediction_decode[i,...]==1]=255
                    img_b[prediction_decode[i,...]==2]=255
                    #img_g[prediction_decode[i,...]==3]=255

                    img_b = remove_small_objects(img_b>0, 50)
                    img_r = remove_small_objects(img_r>0, 100)

                    img_ = np.concatenate((img_b[...,np.newaxis], img_g[...,np.newaxis], img_r[...,np.newaxis]), axis=2)
                    
                    cv2.imwrite(seg_results_small_path+ img_name[i]+ '.png', np.float32(img_)*255)
                    
                    img_ww = cv2.resize(np.float32(img_)*255, (int(ori_width[i]),int(ori_height[i])), interpolation = cv2.INTER_NEAREST)
                    cv2.imwrite(seg_results_raw_path+ img_name[i]+ '.png', img_ww)
                
                
                
                pbar.update(imgs.shape[0])
                
                


if __name__ == '__main__':

    args = parser.parse_args()
    results_path = args.results_path

    """
    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":", 1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        device = torch.device("cuda")
    else:  # cpu
        device = torch.device(args.device)
    """
    device = torch.device("cpu")


    # parse config file if provided
    config_file = args.config_file
    if config_file is not None:
        if not osp.isfile(config_file): raise Exception('non-existent config file')
        with open(args.config_file, 'r') as f:
            args.__dict__ = json.load(f)
    experiment_path = args.experiment_path  # this should exist in a config file
    model_name = args.model_name

    if experiment_path is None: raise Exception('must specify path to experiment')

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    data_path = '../Results/M1/Good_quality/'

    csv_path = 'test_all.csv'
    test_loader = get_test_dataset(data_path, csv_path=csv_path, tg_size=tg_size)
    
    model_1 = get_arch(model_name, n_classes=3).to(device)
    model_2 = get_arch(model_name, n_classes=3).to(device)
    model_3 = get_arch(model_name, n_classes=3).to(device)
    model_4 = get_arch(model_name, n_classes=3).to(device)
    model_5 = get_arch(model_name, n_classes=3).to(device)
    model_6 = get_arch(model_name, n_classes=3).to(device)
    model_7 = get_arch(model_name, n_classes=3).to(device)
    model_8 = get_arch(model_name, n_classes=3).to(device)
    
    experiment_path_1 = './experiments/wnet_All_three_1024_disc_cup/28/'
    experiment_path_2 = './experiments/wnet_All_three_1024_disc_cup/30/'
    experiment_path_3 = './experiments/wnet_All_three_1024_disc_cup/32/'
    experiment_path_4 = './experiments/wnet_All_three_1024_disc_cup/34/'
    experiment_path_5 = './experiments/wnet_All_three_1024_disc_cup/36/'
    experiment_path_6 = './experiments/wnet_All_three_1024_disc_cup/38/'
    experiment_path_7 = './experiments/wnet_All_three_1024_disc_cup/40/'
    experiment_path_8 = './experiments/wnet_All_three_1024_disc_cup/42/'


    model_1, stats = load_model(model_1, experiment_path_1, device)
    model_1.eval()

    model_2, stats = load_model(model_2, experiment_path_2, device)
    model_2.eval()
    
    model_3, stats = load_model(model_3, experiment_path_3, device)
    model_3.eval()
    
    model_4, stats = load_model(model_4, experiment_path_4, device)
    model_4.eval()
    
    model_5, stats = load_model(model_5, experiment_path_5, device)
    model_5.eval()
    
    model_6, stats = load_model(model_6, experiment_path_6, device)
    model_6.eval()
    
    model_7, stats = load_model(model_7, experiment_path_7, device)
    model_7.eval()
    
    model_8, stats = load_model(model_8, experiment_path_8, device)
    model_8.eval()


    prediction_eval(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8, test_loader)
    
    result_path = '../Results/M2/optic_disc_cup/resized/'
    binary_vessel_path = '../Results/M2/binary_vessel/'
    artery_vein_path = '../Results/M2/artery_vein/'
    
    optic_disc_centre(result_path,binary_vessel_path, artery_vein_path)
    
