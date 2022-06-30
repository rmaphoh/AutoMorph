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
Module with endpoints related to server functionality and status.
"""

import flask
from . import app
from . import base_url


@app.route(base_url + "status", methods=["GET"])
def retipy_server_status():
    """
    Simple endpoint that returns 200 if the server is working. Useful to verify if the worker server
    is running.
    :return: HTTP status 200 if the REST server is working.
    """
    return flask.make_response('', 200)
