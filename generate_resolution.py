import pandas as pd
import os
import shutil

raw_img_dir = '/NVME/decrypted/ukbb/fundus/raw/CLRIS/'

if os.path.exists(raw_img_dir+'.ipynb_checkpoints'):
    shutil.rmtree(raw_img_dir+'.ipynb_checkpoints')

image_list = sorted(os.listdir(raw_img_dir))
img_list = []
# import image resolution here
res_list = []

for i in image_list:
    img_list.append(i)
    res_list.append(0.008)
    
Data4stage2 = pd.DataFrame({'fundus':img_list, 'res':res_list})
Data4stage2.to_csv('resolution_information.csv', index = None, encoding='utf8')
