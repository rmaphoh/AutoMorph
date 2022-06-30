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
Script that takes as input an array of tortuosity values and groups them by applying the k-means
algorithm
"""

import argparse
import h5py
import glob
import numpy as np
import os
from collections import Counter
from sklearn.cluster import KMeans

from retipy import configuration

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--configuration",
    help="the configuration file location",
    default="resources/retipy.config")
args = parser.parse_args()

CONFIG = configuration.Configuration(args.configuration)

tags = []
windows = []
for filename in sorted(glob.glob(os.path.join(CONFIG.output_folder, '*.png.h5'))):
    current_file = h5py.File(filename, 'r')
    tag = np.array(current_file['tags'])
    window = np.array(current_file['windows'])
    tags.append(tag)
    windows.append(window)
    current_file.close()

# count how many tortuosity measure we have
tag_count = 0
for tag in tags:
    tag_count += tag.shape[0]
tag_data = np.empty([tag_count, tags[0].shape[1]])

# convert the given array to one single matrix of [tag_count, tortuosity_count]
position = 0
for tag in tags:
    tag_data[position:position + tag.shape[0]] = tag
    position += tag.shape[0]

clustering = KMeans(n_clusters=3, random_state=0).fit_predict(tag_data)

# count how many windows belong to each class
count = Counter(c for c in clustering if c+1)
for cid, count in count.most_common():
    print(cid, count)

# add the clustering evaluation to the end of the tag matrix
the_tag = np.c_[tag_data, clustering]

# create a new matrix to contain all window values
data_count = 0
for window in windows:
    data_count += window.shape[0]
the_window = np.empty([data_count, windows[0].shape[1], windows[0].shape[2], windows[0].shape[3]])

# store the window values in the new matrix
position = 0
for window in windows:
    the_window[position:position+window.shape[0]] = window
    position += window.shape[0]

print(the_window.shape)
print(the_tag.shape)
# save the processed dataset
hf = h5py.File(CONFIG.output_folder + "/input_data.h5", 'w')
hf.create_dataset('windows', data=the_window)
hf.create_dataset('tags', data=the_tag)
hf.close()
