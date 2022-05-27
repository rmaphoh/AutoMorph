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

mkdir -p build
cp -r src/resources build
cp -r condor/* build
cp src/*.py build
cp -r src/retipy build
cp -r src/lib build
cp src/util/configurator.py build
cd build
revision=$(git rev-parse HEAD)
sed -i "s/:latest/:$revision/" retipy.condor
rm setup.py
mv scripts/* .
rm -r scripts
./create_dagman.py
rm create_dagman.py
rm resources/retipy.config
# condor_submit_dag retipy.dag
