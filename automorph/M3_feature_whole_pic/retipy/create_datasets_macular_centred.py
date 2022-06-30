#!/usr/bin/env python3

# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
script to estimate the linear tortuosity of a set of retinal images, it will output the values
to a file in the output folder defined in the configuration. The output will only have the
estimated value and it is sorted by image file name.
"""

def create_dataset_macular_centred():

    import argparse
    import glob
    import os
    import h5py
    import shutil
    import pandas as pd
    
    from .retipy import configuration, retina, tortuosity_measures
    import automorph.config as gv
    from pathlib import Path
    
     
    
    if os.path.exists('{}M2/artery_vein/artery_binary_skeleton/.ipynb_checkpoints'.format(gv.results_dir)):
        shutil.rmtree('{}M2/artery_vein/artery_binary_skeleton/.ipynb_checkpoints'.format(gv.results_dir)) 
    if os.path.exists('{}M2/binary_vessel/binary_skeleton/.ipynb_checkpoints'.format(gv.results_dir)):
        shutil.rmtree('{}M2/binary_vessel/binary_skeleton/.ipynb_checkpoints'.format(gv.results_dir)) 
    if os.path.exists('{}M2/artery_vein/vein_binary_skeleton/.ipynb_checkpoints'.format(gv.results_dir)):
        shutil.rmtree('{}M2/artery_vein/vein_binary_skeleton/.ipynb_checkpoints'.format(gv.results_dir))
    if not os.path.exists('{}M3/Macular_centred/Width/'.format(gv.results_dir)):
        os.makedirs('{}M3/Macular_centred/Width/'.format(gv.results_dir))
    
    #if os.path.exists('./DDR/av_seg/raw/.ipynb_checkpoints'):
    #    shutil.rmtree('./DDR/av_seg/raw/.ipynb_checkpoints') 
    
    cfg_path = Path(__file__).parent / "./resources/retipy.config"
    CONFIG = configuration.Configuration(cfg_path)
    binary_FD_binary,binary_VD_binary,binary_Average_width,binary_t2_list,binary_t4_list,binary_t5_list = [],[],[],[],[],[]
    artery_FD_binary,artery_VD_binary,artery_Average_width,artery_t2_list,artery_t4_list,artery_t5_list = [],[],[],[],[],[]
    vein_FD_binary,vein_VD_binary,vein_Average_width,vein_t2_list,vein_t4_list,vein_t5_list = [],[],[],[],[],[]
    name_list = []
    
    Artery_PATH = '{}M2/artery_vein/macular_centred_artery_skeleton'.format(gv.results_dir)
    Vein_PATH = '{}M2/artery_vein/macular_centred_vein_skeleton'.format(gv.results_dir)
    Binary_PATH = '{}M2/binary_vessel/macular_centred_binary_skeleton'.format(gv.results_dir)
    
    
    for filename in sorted(glob.glob(os.path.join(Binary_PATH, '*.png'))):
        segmentedImage = retina.Retina(None, filename, store_path='{}M2/binary_vessel/macular_centred_binary_process'.format(gv.results_dir))
    
        #segmentedImage.threshold_image()
        #segmentedImage.reshape_square()
        #window_sizes = segmentedImage.get_window_sizes()
        window_sizes = [912]
        window = retina.Window(
            segmentedImage, window_sizes[-1], min_pixels=CONFIG.pixels_per_window)
        FD_binary,VD_binary,Average_width, t2, t4, td = tortuosity_measures.evaluate_window(window, CONFIG.pixels_per_window, CONFIG.sampling_size, CONFIG.r_2_threshold,store_path='{}M2/binary_vessel/macular_centred_binary_process/'.format(gv.results_dir))
        #print(window.tags)
        binary_t2_list.append(t2)
        binary_t4_list.append(t4)
        binary_t5_list.append(td)
        binary_FD_binary.append(FD_binary)
        binary_VD_binary.append(VD_binary)
        binary_Average_width.append(Average_width)
        name_list.append(filename.split('/')[-1])
    
    
    for filename in sorted(glob.glob(os.path.join(Artery_PATH, '*.png'))):
    
        segmentedImage = retina.Retina(None, filename,store_path='{}M2/artery_vein/macular_centred_artery_process'.format(gv.results_dir))
        window_sizes = [912]
        window = retina.Window(
            segmentedImage, window_sizes[-1], min_pixels=CONFIG.pixels_per_window)
        FD_binary,VD_binary,Average_width, t2, t4, td = tortuosity_measures.evaluate_window(window, CONFIG.pixels_per_window, CONFIG.sampling_size, CONFIG.r_2_threshold,store_path='{}M2/artery_vein/macular_centred_artery_process/'.format(gv.results_dir))
        #print(window.tags)
        artery_t2_list.append(t2)
        artery_t4_list.append(t4)
        artery_t5_list.append(td)
        artery_FD_binary.append(FD_binary)
        artery_VD_binary.append(VD_binary)
        artery_Average_width.append(Average_width)
    
    
    for filename in sorted(glob.glob(os.path.join(Vein_PATH, '*.png'))):
    
        segmentedImage = retina.Retina(None, filename,store_path='{}M2/artery_vein/macular_centred_vein_process'.format(gv.results_dir))
        window_sizes = [912]
        window = retina.Window(
            segmentedImage, window_sizes[-1], min_pixels=CONFIG.pixels_per_window)
        FD_binary,VD_binary,Average_width, t2, t4, td = tortuosity_measures.evaluate_window(window, CONFIG.pixels_per_window, CONFIG.sampling_size, CONFIG.r_2_threshold,store_path='{}M2/artery_vein/macular_centred_vein_process/'.format(gv.results_dir))
        #print(window.tags)
        vein_t2_list.append(t2)
        vein_t4_list.append(t4)
        vein_t5_list.append(td)
        vein_FD_binary.append(FD_binary)
        vein_VD_binary.append(VD_binary)
        vein_Average_width.append(Average_width)
        
    Disc_file = pd.read_csv('{}M3/Macular_centred/Disc_cup_results.csv'.format(gv.results_dir))
    
    Data4stage2 = pd.DataFrame({'Fractal_dimension':binary_FD_binary, 'Vessel_density':binary_VD_binary, 'Average_width':binary_Average_width,'Distance_tortuosity':binary_t2_list, 'Squared_curvature_tortuosity':binary_t4_list, 'Tortuosity_density':binary_t5_list, 'Artery_Fractal_dimension':artery_FD_binary, 'Artery_Vessel_density':artery_VD_binary, 'Artery_Average_width':artery_Average_width,'Artery_Distance_tortuosity':artery_t2_list, 'Artery_Squared_curvature_tortuosity':artery_t4_list, 'Artery_Tortuosity_density':artery_t5_list, 'Vein_Fractal_dimension':vein_FD_binary, 'Vein_Vessel_density':vein_VD_binary, 'Vein_Average_width':vein_Average_width,'Vein_Distance_tortuosity':vein_t2_list, 'Vein_Squared_curvature_tortuosity':vein_t4_list, 'Vein_Tortuosity_density':vein_t5_list})
    #Data4stage2 = pd.concat([Disc_file[Disc_file['Name'].isin(name_list)].reset_index().drop(columns=['index', 'Name']), Data4stage2] ,axis=1)
    Data4stage2 = pd.concat([Disc_file[Disc_file['Name'].isin(name_list)].reset_index(), Data4stage2] ,axis=1)
    Data4stage2.to_csv('{}M3/Macular_centred/Macular_Measurement.csv'.format(gv.results_dir), index = None, encoding='utf8')
