import fundus_prep as prep
import os
import pandas as pd
from PIL import ImageFile
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process(image_list, save_path):
    
    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    list_resolution = []
    scale_resolution = []
    
    resolution_list = pd.read_csv('resolution_information.csv')
    raw_img_dir = '/NVME/decrypted/ukbb/fundus/raw/CLRIS/'
    crop_img_dir = '/NVME/decrypted/ukbb/fundus/raw/CLRIS_cropped/'
    
    for image_path in image_list:
        
        dst_image = raw_img_dir + image_path
        if os.path.exists(crop_img_dir + image_path):
            print('continue...')
            continue
        try:
            resolution_ = resolution_list['res'][resolution_list['fundus']==image_path].values[0]
            list_resolution.append(resolution_)
            img = prep.imread(dst_image)
            r_img, borders, mask, r_img, radius_list,centre_list_w, centre_list_h = prep.process_without_gb(img,img,radius_list,centre_list_w, centre_list_h)
            prep.imwrite(save_path + image_path.split('.')[0] + '.png', r_img)
            name_list.append(image_path.split('.')[0] + '.png')
        
        except:
            pass

    scale_list = [a*2/912 for a in radius_list]
    scale_resolution = [a*b*1000 for a,b in zip(list_resolution,scale_list)]
    Data4stage2 = pd.DataFrame({'Name':name_list, 'centre_w':centre_list_w, 'centre_h':centre_list_h, 'radius':radius_list, 'Scale':scale_list, 'Scale_resolution':scale_resolution})
    Data4stage2.to_csv('/NVME/decrypted/scratch/ukbb_fundus_crop/crop_info_run2.csv', index = None, encoding='utf8')
    
    print('END OF SCRIPT')


if __name__ == "__main__":
    if os.path.exists('/NVME/decrypted/ukbb/fundus/raw/CLRIS/.ipynb_checkpoints'):
        shutil.rmtree('/NVME/decrypted/ukbb/fundus/raw/CLRIS/.ipynb_checkpoints')
    image_list = sorted(os.listdir('/NVME/decrypted/ukbb/fundus/raw/CLRIS/'))
    save_path = '/NVME/decrypted/ukbb/fundus/raw/CLRIS_cropped/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    process(image_list, save_path)

