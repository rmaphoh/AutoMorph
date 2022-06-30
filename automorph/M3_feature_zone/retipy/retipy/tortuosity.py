# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017-2018  Alejandro Valdes
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
Module that applies tortuosity measures to given images. It used the measures defined in
tortuosity_measures
"""

import numpy as np
from retipy import retina, tortuosity_measures


def _tortuosity_window(x1: int, y1: int, x2: int, y2: int, notes: str):
    tw = {}
    tw["notes"] = notes
    tw["x"] = [x1, x1, x2, x2]
    tw["y"] = [y1, y2, y2, y1]
    return tw


def density(
        image: np.ndarray,
        window_size: int = 10,
        min_pixels: int = 10,
        creation_method: str = "separated",
        threshold: float = 0.97) -> dict:
    image = retina.Retina(image, "tortuosity_density")
    dimension = image.reshape_by_window(window_size, True)
    image.threshold_image()
    image.skeletonization()
    windows = retina.Window(image, dimension, min_pixels=min_pixels, method=creation_method)
    evaluation = \
        {
            "uri": "tortuosity_density",
            "data": [],
            # "image": image.original_base64  # TODO: maybe return a processed image?
        }

    for i in range(0, windows.shape[0]):
        window = windows.windows[i, 0]
        w_pos = windows.w_pos[i]
        image = retina.Retina(window, "td")

        vessels = retina.detect_vessel_border(image)
        processed_vessel_count = 0
        for vessel in vessels:
            if len(vessel[0]) > 10:
                processed_vessel_count += 1
                tortuosity_density = tortuosity_measures.tortuosity_density(vessel[0], vessel[1])
                if tortuosity_density > threshold:
                    evaluation["data"].append(_tortuosity_window(
                        w_pos[0, 0].item(),
                        w_pos[0, 1].item(),
                        w_pos[1, 0].item(),
                        w_pos[1, 1].item(),
                        "{0:.2f}".format(tortuosity_density)))

    return evaluation


def fractal(
        image: np.ndarray,
        window_size: int = 10,
        min_pixels: int = 56,
        creation_method: str = "separated",
        threshold: float = 0.94) -> dict:
    image = retina.Retina(image, "tortuosity_density")
    dimension = image.reshape_by_window(window_size, True)
    image.threshold_image()
    image.skeletonization()
    windows = retina.Window(image, dimension, min_pixels=min_pixels, method=creation_method)
    evaluation = \
        {
            "uri": "fractal_dimension",
            "data": [],
            # "image": image.original_base64  # TODO: maybe return a processed image?
        }

    for i in range(0, windows.shape[0]):
        window = windows.windows[i, 0]
        w_pos = windows.w_pos[i]
        image = retina.Retina(window, "tf")
        vessels = retina.detect_vessel_border(image)
        processed_vessel_count = 0
        for vessel in vessels:
            if len(vessel[0]) > 10:
                processed_vessel_count += 1
                fractal_tortuosity = tortuosity_measures.fractal_tortuosity_curve(vessel[0], vessel[1])
                if fractal_tortuosity > threshold:
                    evaluation["data"].append(_tortuosity_window(
                        w_pos[0, 0].item(),
                        w_pos[0, 1].item(),
                        w_pos[1, 0].item(),
                        w_pos[1, 1].item(),
                        "{0:.2f}".format(fractal_tortuosity)))

    return evaluation
