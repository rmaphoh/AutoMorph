from glob import glob
from sys import exit
import os
import cv2 as cv
import pandas as pd
from PIL import ImageFile
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True
import automorph.config as gv 
from automorph.M0_Preprocess import fundus_prep as prep
from random import sample
from pathlib import Path

def create_resolution_information():
    # create resolution of 1 for all images if information not known.

    images = glob("{}*.png".format(gv.image_dir))
    print("{} images found with glob".format(len(images)))
    
    res_csv_pth = Path(__file__).parent / "../resolution_information.csv" 
    with open(res_csv_pth, "w") as f:
      f.write("fundus,res\n")
      f.writelines("{},1\n".format(x.split('/')[-1]) for x in images)  	

def process(image_list, save_path):
    
    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    list_resolution = []
    scale_resolution = []
    
    resolution_csv_path = Path(__file__).parent / "../resolution_information.csv"
    if not os.path.exists(resolution_csv_path):
        create_resolution_information()
    resolution_list = pd.read_csv(resolution_csv_path)
   
    for image_path in image_list:
        
        dst_image = gv.image_dir + image_path
        if os.path.exists('{}M0/images/'.format(save_path) + image_path):
            print('continue...')
            continue
            
        try:
            if len(resolution_list['res'][resolution_list['fundus']==image_path].values) == 0:
                resolution_ = 0
            else:
                resolution_ = resolution_list['res'][resolution_list['fundus']==image_path].values[0]
            list_resolution.append(resolution_)

            img = prep.imread(dst_image)
            r_img, borders, mask, r_img, radius_list,centre_list_w, centre_list_h = prep.process_without_gb(img,img,radius_list,centre_list_w, centre_list_h)
            prep.imwrite(save_path + image_path.split('.')[0] + '.png', r_img)
            #prep.imwrite('../Results/M0/images/' + image_path.split('.')[0] + '.png', mask)
        except IndexError:
            print("\nThe file {} has not been added to the resolution_information.csv found at {}\n\
                   Please update this file with the script found at /lee_lab_scripts/create_resolution.py and re-run the code".format( \
                       image_path, resolution_csv_path))
            exit(1)

        name_list.append(image_path.split('.')[0] + '.png')
        

    scale_list = [a*2/912 for a in radius_list]
    
    scale_resolution = [a*b*1000 for a,b in zip(list_resolution,scale_list)]
    
    Data4stage2 = pd.DataFrame({'Name':name_list, 'centre_w':centre_list_w, 'centre_h':centre_list_h, 'radius':radius_list, 'Scale':scale_list, 'Scale_resolution':scale_resolution})
    Data4stage2.to_csv('{}M0/crop_info.csv'.format(gv.results_dir), index = None, encoding='utf8')
        

def EyeQ_process():

    if gv.sample_num:
        print("Sampling {} images from {}".format(gv.sample_num, gv.image_dir))
        image_list = sample(sorted(os.listdir(gv.image_dir)), gv.sample_num)        
    else:
        image_list = sorted(os.listdir(gv.image_dir))

    save_path = gv.results_dir + "M0/images/"

    if not os.path.exists('{}'.format(save_path)):
        os.makedirs('{}'.format(save_path))
 
    process(image_list, save_path)

if __name__ == '__main__':
    EyeQ_process()