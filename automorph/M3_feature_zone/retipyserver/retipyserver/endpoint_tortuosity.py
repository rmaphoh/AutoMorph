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
Module with all REST endpoints to calculate tortuosity measures given a grayscale image
All operations are implemented as POST
"""

import base64
import flask
import io
import numpy as np
from PIL import Image
from retipy import tortuosity
from . import app
from . import base_url

tortuosity_url = base_url + "tortuosity/"


@app.route(tortuosity_url + "density", methods=["POST"])
def post_tortuosity_density():
    data = {"success": False}

    if flask.request.method == "POST":
        json = flask.request.get_json(silent=True)
        if json is not None:  # pragma: no cover
            image = base64.b64decode(json["image"])
            image = Image.open(io.BytesIO(image)).convert('L')
            data = tortuosity.density(np.array(image))
    return flask.jsonify(data)


@app.route(tortuosity_url + "fractal", methods=["POST"])
def post_tortuosity_fractal():
    data = {"success": False}

    if flask.request.method == "POST":
        json = flask.request.get_json(silent=True)
        if json is not None:  # pragma: no cover
            image = base64.b64decode(json["image"])
            image = Image.open(io.BytesIO(image)).convert('L')
            data = tortuosity.fractal(np.array(image))
    return flask.jsonify(data)
