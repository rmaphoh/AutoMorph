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

import argparse
import glob
# import numpy as np
import os
import h5py
import shutil
import pandas as pd
# import scipy.stats as stats

from retipy import configuration, retina, tortuosity_measures

AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA','../..')

if os.path.exists('/home/jupyter/Deep_rias/Results/M2/artery_vein/artery_binary_skeleton/.ipynb_checkpoints'):
    shutil.rmtree('/home/jupyter/Deep_rias/Results/M2/artery_vein/artery_binary_skeleton/.ipynb_checkpoints') 
if os.path.exists('/home/jupyter/Deep_rias/Results/M2/binary_vessel/binary_skeleton/.ipynb_checkpoints'):
    shutil.rmtree('/home/jupyter/Deep_rias/Results/M2/binary_vessel/binary_skeleton/.ipynb_checkpoints') 
if os.path.exists('/home/jupyter/Deep_rias/Results/M2/artery_vein/vein_binary_skeleton/.ipynb_checkpoints'):
    shutil.rmtree('/home/jupyter/Deep_rias/Results/M2/artery_vein/vein_binary_skeleton/.ipynb_checkpoints')
if not os.path.exists(f'{AUTOMORPH_DATA}/Results/M3/Width/'):
    os.makedirs(f'{AUTOMORPH_DATA}/Results/M3/Width/')

#if os.path.exists('./DDR/av_seg/raw/.ipynb_checkpoints'):
#    shutil.rmtree('./DDR/av_seg/raw/.ipynb_checkpoints') 


parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--configuration",
    help="the configuration file location",
    default="resources/retipy.config")
args = parser.parse_args()

CONFIG = configuration.Configuration(args.configuration)
t2_list = []
t4_list = []
t5_list = []
name_list = []

Artery_PATH = '/home/jupyter/Deep_rias/Results/M2/artery_vein/artery_binary_skeleton'
Vein_PATH = '/home/jupyter/Deep_rias/Results/M2/artery_vein/vein_binary_skeleton'


for filename in sorted(glob.glob(os.path.join(CONFIG.image_directory, '*.png'))):
    segmentedImage = retina.Retina(None, filename, store_path='/home/jupyter/Deep_rias/Results/M2/binary_vessel/binary_process')
    #segmentedImage.threshold_image()
    #segmentedImage.reshape_square()
    #window_sizes = segmentedImage.get_window_sizes()
    window_sizes = [912]
    window = retina.Window(
        segmentedImage, window_sizes[-1], min_pixels=CONFIG.pixels_per_window)
    t1, t2, t3, t4, td, tfi, tft, vessel_density, average_caliber,vessel_count,tcurve, bifurcation_t, vessel_count_1, vessel_count_list, w1_list = tortuosity_measures.evaluate_window(window, CONFIG.pixels_per_window, CONFIG.sampling_size, CONFIG.r_2_threshold,store_path='/home/jupyter/Deep_rias/Results/M2/binary_vessel/binary_process/')
    #print(window.tags)
    t2_list.append(t2)
    t4_list.append(t4)
    t5_list.append(td)
    name_list.append(filename.split('/')[-1])
    #hf = h5py.File(CONFIG.output_folder + "/" + segmentedImage.filename + ".h5", 'w')
    #hf.create_dataset('windows', data=window.windows)
    #hf.create_dataset('tags', data=window.tags)
    #hf.close()
    Data4stage2 = pd.DataFrame({'Order':vessel_count_list, 'Width':w1_list})
    Data4stage2.to_csv(f'{AUTOMORPH_DATA}/Results/M3/Width/width_results_{segmentedImage._file_name}.csv', index = None, encoding='utf8')

Exit_file = pd.read_csv(f'{AUTOMORPH_DATA}/Results/M3/Binary_Features_Measurement.csv')
Data4stage2 = pd.DataFrame({'Distance_Tortuosity':t2_list, 'Squared_Curvature_Tortuosity':t4_list, 'Tortuosity_density':t5_list})
Data4stage2 = pd.concat([Exit_file, Data4stage2], axis=1)
Data4stage2.to_csv(f'{AUTOMORPH_DATA}/Results/M3/Binary_Tortuosity_Measurement.csv', index = None, encoding='utf8')


    
####################################3
t2_list = []
t4_list = []
t5_list = []
name_list = []

for filename in sorted(glob.glob(os.path.join(Artery_PATH, '*.png'))):

    segmentedImage = retina.Retina(None, filename,store_path='/home/jupyter/Deep_rias/Results/M2/artery_vein/artery_binary_process')
    #segmentedImage.threshold_image()
    #segmentedImage.reshape_square()
    #window_sizes = segmentedImage.get_window_sizes()
    window_sizes = [912]
    window = retina.Window(
        segmentedImage, window_sizes[-1], min_pixels=CONFIG.pixels_per_window)
    t1, t2, t3, t4, td, tfi, tft, vessel_density, average_caliber,vessel_count,tcurve, bifurcation_t, vessel_count_1, vessel_count_list, w1_list = tortuosity_measures.evaluate_window(window, CONFIG.pixels_per_window, CONFIG.sampling_size, CONFIG.r_2_threshold,store_path='/home/jupyter/Deep_rias/Results/M2/artery_vein/artery_binary_process/')
    #print(window.tags)
    t2_list.append(t2)
    t4_list.append(t4)
    t5_list.append(td)
    name_list.append(filename.split('/')[-1])
    #hf = h5py.File(CONFIG.output_folder + "/" + segmentedImage.filename + ".h5", 'w')
    #hf.create_dataset('windows', data=window.windows)
    #hf.create_dataset('tags', data=window.tags)
    #hf.close()
    print(filename.split('/')[-1])
    Data4stage2 = pd.DataFrame({'Order':vessel_count_list, 'Width':w1_list})
    Data4stage2.to_csv(f'{AUTOMORPH_DATA}/Results/M3/Width/artery_width_results_{segmentedImage._file_name}.csv', index = None, encoding='utf8')
    
Exit_file = pd.read_csv(f'{AUTOMORPH_DATA}/Results/M3/Artery_Features_Measurement.csv')    
Data4stage2 = pd.DataFrame({'Tortuosity':t2_list, 'Squared_Curvature_Tortuosity':t4_list, 'Tortuosity_density':t5_list})
Data4stage2 = pd.concat([Exit_file, Data4stage2], axis=1)
Data4stage2.to_csv(f'{AUTOMORPH_DATA}/Results/M3/Artery_Tortuosity_Measurement.csv', index = None, encoding='utf8')

####################################3
t2_list = []
t4_list = []
t5_list = []
name_list = []

for filename in sorted(glob.glob(os.path.join(Vein_PATH, '*.png'))):

    segmentedImage = retina.Retina(None, filename,store_path='/home/jupyter/Deep_rias/Results/M2/artery_vein/vein_binary_process')
    #segmentedImage.threshold_image()
    #segmentedImage.reshape_square()
    #window_sizes = segmentedImage.get_window_sizes()
    window_sizes = [912]
    window = retina.Window(
        segmentedImage, window_sizes[-1], min_pixels=CONFIG.pixels_per_window)
    t1, t2, t3, t4, td, tfi, tft, vessel_density, average_caliber,vessel_count,tcurve, bifurcation_t, vessel_count_1, vessel_count_list, w1_list = tortuosity_measures.evaluate_window(window, CONFIG.pixels_per_window, CONFIG.sampling_size, CONFIG.r_2_threshold,store_path='/home/jupyter/Deep_rias/Results/M2/artery_vein/vein_binary_process/')
    #print(window.tags)
    t2_list.append(t2)
    t4_list.append(t4)
    t5_list.append(td)
    name_list.append(filename.split('/')[-1])
    #hf = h5py.File(CONFIG.output_folder + "/" + segmentedImage.filename + ".h5", 'w')
    #hf.create_dataset('windows', data=window.windows)
    #hf.create_dataset('tags', data=window.tags)
    #hf.close()
    Data4stage2 = pd.DataFrame({'Order':vessel_count_list, 'Width':w1_list})
    Data4stage2.to_csv(f'{AUTOMORPH_DATA}/Results/M3/Width/vein_width_results_{segmentedImage._file_name}.csv', index = None, encoding='utf8')
    
Exit_file = pd.read_csv(f'{AUTOMORPH_DATA}/Results/M3/Vein_Features_Measurement.csv') 
Data4stage2 = pd.DataFrame({'Image_id':name_list, 'Tortuosity':t2_list, 'Squared_Curvature_Tortuosity':t4_list, 'Tortuosity_density':t5_list})
Data4stage2 = pd.concat([Exit_file, Data4stage2], axis=1)
Data4stage2.to_csv(f'{AUTOMORPH_DATA}/Results/M3/Vein_Tortuosity_Measurement.csv', index = None, encoding='utf8')

