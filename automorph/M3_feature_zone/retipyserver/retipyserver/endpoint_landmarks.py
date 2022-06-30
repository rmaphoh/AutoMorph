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
Module with all REST endpoints to identify and classify landmarks given a grayscale image
All operations are implemented as POST
"""

import base64
import flask
import io
import numpy as np
from PIL import Image
from retipy import landmarks
from . import app
from . import base_url

landmarks_url = base_url + "landmarks/"


@app.route(landmarks_url + "classification", methods=["POST"])
def post_landmarks_classification():
    data = {"success": False}

    if flask.request.method == "POST":
        json = flask.request.get_json(silent=True)
        if json is not None:  # pragma: no cover
            image = base64.b64decode(json["image"])
            image = Image.open(io.BytesIO(image)).convert('L')
            bifurcations_data, crossings_data = landmarks.classification(np.array(image), 20)
            data = {"bifurcations": bifurcations_data, "crossings": crossings_data}
    return flask.jsonify(data)
