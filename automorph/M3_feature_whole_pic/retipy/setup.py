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

from setuptools import setup, find_packages

setup(
    name="retipy",
    version="0.0.1.dev0",
    description="retinal image processing on python",
    author="Alejandro Valdes",
    author_email="alejandrovaldes@live.com",
    license="GPLv3",
    python="image-processing python retina",
    url="https://github.com/alevalv/retipy",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7'
    ],
    packages=find_packages(exclude=["test", "util"]),
    install_requires=['matplotlib', 'numpy', 'pillow', 'scikit-image', 'scipy', 'h5py', 'scikit-learn', 'tensorflow', 'keras']
    )
