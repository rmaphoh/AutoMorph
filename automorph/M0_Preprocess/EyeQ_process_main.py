from genericpath import isfile
from glob import glob
import pandas as pd
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .fundus_prep import process_without_gb, imread, imwrite
from random import sample
from pathlib import Path


def create_resolution_information(image_dir):
    # create resolution of 1 for all images if information not known.

    images = glob("{}*.png".format(image_dir))
    print("{} images found with glob".format(len(images)))
    
    res_csv_pth = Path(__file__).parent / "../resolution_information.csv" 
    with open(res_csv_pth, "w") as f:
      f.write("fundus,res\n")
      f.writelines("{},1\n".format(x.split('/')[-1]) for x in images)  	

def process(image_list, save_path, cfg):
    """ Crops each image in the image list to create the smallest square that fits all retinal colored information and 
    removes background retina pixels
    """
    
    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    list_resolution = []
    scale_resolution = []
    
    resolution_csv_path = Path(__file__).parent / "../resolution_information.csv"
    if not os.path.exists(resolution_csv_path):
        create_resolution_information(cfg.image_dir)
    resolution_list = pd.read_csv(resolution_csv_path)
   
    for image_path in image_list:
        
        dst_image = cfg.image_dir + image_path
        if os.path.exists('{}M0/images/'.format(save_path) + image_path):
            print('continue...')
            continue
        try:         
            if len(resolution_list['res'][resolution_list['fundus']==image_path].values) == 0:
                resolution_ = 1
            else:
                resolution_ = resolution_list['res'][resolution_list['fundus']==image_path].values[0]
            list_resolution.append(resolution_)
    
            img = imread(dst_image)
            r_img, borders, mask, label, radius_list,centre_list_w, centre_list_h = process_without_gb(img,img,radius_list,centre_list_w, centre_list_h)

            if not cfg.sparse:
                imwrite(save_path + image_path.split('.')[0] + '.png', r_img)

        except IndexError:
            print("\nThe file {} has not been added to the resolution_information.csv found at {}\n\
                   Please update this file with the script found at /lee_lab_scripts/create_resolution.py and re-run the code".format( \
                       image_path, resolution_csv_path))
            exit(1)
        except ValueError:
            print('error with boundary for')

        name_list.append(cfg.image_dir+image_path)
        

    scale_list = [a*2/912 for a in radius_list]
    
    scale_resolution = [a*b*1000 for a,b in zip(list_resolution,scale_list)]
    
    Data4stage2 = pd.DataFrame({'Name':name_list, 'centre_w':centre_list_w, 'centre_h':centre_list_h, 'radius':radius_list, 'Scale':scale_list, 'Scale_resolution':scale_resolution})
    Data4stage2.to_csv('{}M0/crop_info.csv'.format(cfg.results_dir), index = None, encoding='utf8')
        

def EyeQ_process(cfg):

    if cfg.sample_num:
        print("Sampling {} images from {}".format(cfg.sample_num, cfg.image_dir))
        image_list = sample(sorted(os.listdir(cfg.image_dir)), cfg.sample_num)        
    else:
        image_list = sorted(os.listdir(cfg.image_dir))

    save_path = cfg.results_dir + "M0/images/"

    if not os.path.exists('{}'.format(save_path)):
        os.makedirs('{}'.format(save_path))
 
    process(image_list, save_path, cfg)