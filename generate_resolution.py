import pandas as pd
import os
import sys
import shutil

AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA','.')

# read pixel_resolution from cli arg if defined, otherwise use default value 0.008
pixel_resolution = float(sys.argv[1]) if len(sys.argv) > 1 else 0.008

if os.path.exists(f'{AUTOMORPH_DATA}/images/.ipynb_checkpoints'):
    shutil.rmtree(f'{AUTOMORPH_DATA}/images/.ipynb_checkpoints')

image_list = sorted(os.listdir(f'{AUTOMORPH_DATA}/images/'))
img_list = []
# import image resolution here
res_list = []

for i in image_list:
    img_list.append(i)
    res_list.append(pixel_resolution)
    
Data4stage2 = pd.DataFrame({'fundus':img_list, 'res':res_list})
Data4stage2.to_csv(f'{AUTOMORPH_DATA}/resolution_information.csv', index = None, encoding='utf8')
