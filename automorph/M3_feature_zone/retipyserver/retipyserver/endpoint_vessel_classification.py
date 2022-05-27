# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Maria Aguiar
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
Module with all REST endpoints to classify blood vessels into arteries and veins given a original image, segmented image
and a/v image
All operations are implemented as POST
"""

import base64
import flask
import io
import numpy as np
from PIL import Image
from retipy import vessel_classification
from retipy.retina import Retina
from . import app
from . import base_url

vessel_classification_url = base_url + "vessel_classification/"


@app.route(vessel_classification_url + "classification", methods=["POST"])
def post_vessel_classification():
    data = {"success": False}

    if flask.request.method == "POST":
        json = flask.request.get_json(silent=True)
        if json is not None:  # pragma: no cover
            segmented = base64.b64decode(json["segmented_image"])
            segmented = Image.open(io.BytesIO(segmented)).convert('L')
            original = base64.b64decode(json["original_image"])
            original = Image.open(io.BytesIO(original)).convert('RGB')
            data = {
                "classification": Retina.get_base64_image(
                    vessel_classification.classification(
                        np.array(original), np.array(segmented)),
                    False)}
    return flask.jsonify(data)
