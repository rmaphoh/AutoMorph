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

import argparse
import re
import sys
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument("-fd", "--file-directory", help="directory to parse info")
parser.add_argument("-e", "--extension", help="extension of the files to parse")

args = parser.parse_args()
directory = args.file_directory

files = [f for f in listdir(directory) if (isfile(join(directory, f)) and f.endswith(args.extension))]

output = ""
S = ','
empty_files = 0
for file in files:
    content = open(directory + '/' + file)
    current_image = 1
    parsed_file = re.search('(\d+)-(\d+)-(\d+)-(0\.\d+)' + args.extension, file)
    w = parsed_file.group(1)
    ppw = parsed_file.group(2)
    ss = parsed_file.group(3)
    r2t = parsed_file.group(4)
    for line in content:
        if line:
            print("{:02d},{},{},{},{},{}".format(
                current_image, w, ppw, ss, r2t, line))
            current_image += 1
    empty_files += 20 - min(20, current_image)

print("missing {} image values".format(empty_files),  file=sys.stderr)
