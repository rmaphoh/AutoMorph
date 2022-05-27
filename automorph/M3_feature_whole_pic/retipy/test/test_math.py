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
from retipy import math


class TestMath(TestCase):
    def test_derivative1_forward_h2(self):
        self.assertEqual(
            math.derivative1_forward_h2(0, [1, 2, 3]),
            1.0,
            "first derivative does not match")

    def test_derivative1_forward_h2_error(self):
        self.assertRaises(ValueError, math.derivative1_forward_h2, 0, [])

    def test_derivative1_centered_h1(self):
        self.assertEqual(
            math.derivative1_centered_h1(1, [1, 2, 3]),
            1.0,
            "first derivative does not match")

    def test_derivative1_centered_h1_error(self):
        self.assertRaises(ValueError, math.derivative1_centered_h1, 0, [])

    def test_derivative2_centered_h1(self):
        self.assertEqual(math.derivative2_centered_h1(1, [1, 2, 3]), 0, "second derivative does not match")

    def test_derivative2_centered_h1_error(self):
        self.assertRaises(ValueError, math.derivative2_centered_h1, 0, [])
