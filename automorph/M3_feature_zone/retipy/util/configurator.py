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
Helper script to create configuration files by specifying parameters
Used by condor scripts to test a range of configurations easily.
"""

import argparse
import configparser
from retipy import configuration

parser = argparse.ArgumentParser()
#  parser.add_argument("-v", "--verbose", help="increases retipy output messages", action="store_true")
parser.add_argument("-p", "--path")
parser.add_argument("-w", "--window", help="set the window size manually")
parser.add_argument("-id", "--image-directory", help="specify the directory where images are located")
parser.add_argument(
    "-ppw", "--pixels-per-window", help="sets how many pixels with value are necessary to select a window")
parser.add_argument("-ss", "--sampling-size", help="how many samples are extracted when checking tortuosity")
parser.add_argument(
    "-r2t", "--r2-threshold", help="sets the determination coefficient to determine if a curve is tortuous")
parser.add_argument("-o", "--output-folder", help="sets the output folder to print image results")

args = parser.parse_args()

test_configuration = configparser.ConfigParser()
test_configuration[configuration.PROPERTY_DEFAULT_CATEGORY] = {
    configuration.PROPERTY_IMAGE_DIRECTORY: args.image_directory,
    configuration.PROPERTY_WINDOW_SIZE: args.window,
    configuration.PROPERTY_PIXELS_PER_WINDOW: args.pixels_per_window,
    configuration.PROPERTY_SAMPLING_SIZE: args.sampling_size,
    configuration.PROPERTY_R2_THRESHOLD: args.r2_threshold,
    configuration.PROPERTY_OUTPUT_FOLDER: args.output_folder
}
with open(args.path, 'w') as configfile:
    test_configuration.write(configfile)
