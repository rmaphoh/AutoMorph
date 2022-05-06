#!/usr/bin/env python3
# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Alejandro Valdes
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
Executable file to test some tortuosity measures included in the tortuosity module of retipy
"""

import argparse
import json
import numpy as np
from PIL import Image
from retipy import tortuosity


parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--algorithm", default="TD", help="the tortuosity algorithm to apply")
parser.add_argument(
    "-t", "--threshold", default=0.5, type=float, help="threshold to consider a window as tortuous")
parser.add_argument("-i", "--image-path", help="the path to the retinal image to be processed")
parser.add_argument("-w", "--window-size", default=56, type=int, help="the window size")
parser.add_argument(
    "-wcm",
    "--window-creation-method",
    default="combined",
    help="the window creation mode, can be separated or combined")

args = parser.parse_args()

# TODO: this should be able to process from a basic test, a RBG image, right now it will be on segmentation only
image = Image.open('../images/007-2489-100.png').convert('L')
evaluation = {"success": False}

if args.algorithm == "TD":
    evaluation = tortuosity.density(
        np.array(image),
        window_size=args.window_size,
        min_pixels=10,
        creation_method=args.window_creation_method,
        threshold=args.threshold)

encoder = json.JSONEncoder()
print(encoder.encode(evaluation))
