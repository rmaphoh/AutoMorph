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

"""tests for landmarks endpoint module"""

import json
import sys
from retipy.retina import Retina
from retipyserver import app
from unittest import TestCase


class TestLandmarksEndpoint(TestCase):
    _resources = 'retipy/resources/images/'
    _image_file_name = 'img02.png'
    _image_path = _resources + _image_file_name

    def setUp(self):

        self.image = Retina(None, self._image_path).original_base64
        self.app = app.test_client()

    def test_classification_no_success(self):
        response = self.app.post("/retipy/landmarks/classification")
        self.assertEqual(json.loads(response.get_data().decode(sys.getdefaultencoding())), {'success': False})