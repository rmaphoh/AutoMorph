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

"""tests for tortuosity module"""

from unittest import TestCase
from retipy.retina import Retina
from retipy import tortuosity as t


class TestTortuosity(TestCase):
    _resources = 'retipy/resources/images/'
    _image_file_name = 'img01.png'
    _image_path = _resources + _image_file_name

    def setUp(self):
        self.image = Retina(None, self._image_path)

    def test_density(self):
        result = t.density(self.image.np_image)
        self.assertEqual(result["uri"], "tortuosity_density", "uri does not match")
        self.assertEqual(len(result["data"]), 3, "data size does not match")

    def test_fractal(self):
        result = t.fractal(self.image.np_image)
        self.assertEqual(result["uri"], "fractal_dimension", "uri does not match")
        self.assertEqual(len(result["data"]), 16, "data size does not match")
