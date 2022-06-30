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
# import scipy.stats as stats

from retipy import configuration, retina, tortuosity_measures

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--configuration",
    help="the configuration file location",
    default="resources/retipy.config")
args = parser.parse_args()

CONFIG = configuration.Configuration(args.configuration)

for filename in sorted(glob.glob(os.path.join(CONFIG.image_directory, '*.png'))):
    print("processing {}...".format(filename))
    segmentedImage = retina.Retina(None, filename)
    segmentedImage.threshold_image()
    segmentedImage.reshape_square()
    window_sizes = segmentedImage.get_window_sizes()
    window = retina.Window(
        segmentedImage, window_sizes[-1], min_pixels=CONFIG.pixels_per_window)
    tortuosity_measures.evaluate_window(window, CONFIG.pixels_per_window, CONFIG.sampling_size, CONFIG.r_2_threshold)
    hf = h5py.File(CONFIG.output_folder + "/" + segmentedImage.filename + ".h5", 'w')
    hf.create_dataset('windows', data=window.windows)
    hf.create_dataset('tags', data=window.tags)
    hf.close()
