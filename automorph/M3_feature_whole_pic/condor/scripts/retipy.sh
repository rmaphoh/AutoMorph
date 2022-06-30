#!/bin/sh

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

set -e

# create the folder test if it doesn't exist
if [ ! -d "output" ]; then
    mkdir output
fi

# create the configuration file for this execution, using the passed parameters
./configurator.py -p $1 -id $2 -o $3 -w $4 -ppw $5 -ss $6 -r2t $7

# execute the algorithm
./x_tortuosity.py -c $1 -a fractal
