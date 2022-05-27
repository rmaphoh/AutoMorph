# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Luis Felipe Castaño
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
Module with all REST endpoints to calculate tortuosity measures given a grayscale image
All operations are implemented as POST
"""

import base64
import io
import flask
import numpy as np
from PIL import Image
import cv2
from retipy import retina_grayscale
from . import app
from . import base_url

segmentation_url = base_url + "segmentation/"

@app.route(segmentation_url + "double_segmentation", methods=["POST"])
def post_segmentation_double_segmentation():
    data = {"success": False} # pragma: no cover

    if flask.request.method == "POST": # pragma: no cover
        json = flask.request.get_json(silent=True)
        if json is not None:  # pragma: no cover
            image = base64.b64decode(json["image"])
            image = Image.open(io.BytesIO(image))
            retina = retina_grayscale.Retina_grayscale(np.array(image), None)
            data = {"segmentation": retina.double_segmentation()}
    return flask.jsonify(data) # pragma: no cover
