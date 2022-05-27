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

"""tests for tortuosity measures module"""

from unittest import TestCase
from numpy.testing import assert_array_equal

from retipy import tortuosity_measures as tm, retina


class TestTortuosityMeasures(TestCase):
    _straight_line =[1, 2, 3, 4, 5, 6, 7]

    tm.SAMPLING_SIZE = 2

    def test_linear_regression_tortuosity(self):
        self.assertEqual(
            tm.linear_regression_tortuosity(self._straight_line, self._straight_line),
            1,
            "a straight line should return 1")

    def test_linear_regression_tortuosity_error_size(self):
        self.assertRaises(
            ValueError, tm.linear_regression_tortuosity, [1], [1])

    def test_linear_regression_tortuosity_no_second_try(self):
        self.assertEqual(
            tm.linear_regression_tortuosity([1, 2, 3, 4], [1, 1, 1, 1], False),
            1,
            "should return 1")

    def test_linear_regression_tortuosity_no_interpolation(self):
        self.assertEqual(
            tm.linear_regression_tortuosity([1, 2, 3, 4], [1, 1, 1, 1]),
            1,
            "should return 1")

    def test_distance_2p(self):
        self.assertEqual(tm._distance_2p(0, 0, 0, 1), 1, "distance does not match")

    def test_curve_length(self):
        self.assertEqual(
            tm._curve_length([0, 0], [0, 1]), 1, "curve distance does not match")

    def test_distance_measure_tortuosity(self):
        self.assertEqual(
            tm.distance_measure_tortuosity([0, 2, 4], [0, 2, 4]),
            1,
            "distance measure does not match")

    def test_distance_measure_tortuosity_error(self):
        self.assertRaises(
            ValueError, tm.distance_measure_tortuosity, [1], [2])

    def test_detect_inflection_points(self):
        assert_array_equal(
            [2, 3, 4],
            tm._detect_inflection_points([0, 1, 2, 3, 4, 5], [4, 6, 8, 6, 9, 0]),
            "inflection points does not match")

    def test_distance_inflection_count_tortuosity(self):
        self.assertEqual(
            tm.distance_inflection_count_tortuosity([0, 2, 4], [0, 2, 4]),
            1,
            "inflection count tortuosity value does not match")

    def test_fractal_tortuosity(self):
        self.assertAlmostEqual(
            tm.fractal_tortuosity(retina.Retina(None, "retipy/resources/images/img01.png")),
            1.703965,
            msg="fractal tortuosity does not match",
            delta=0.00001)

    def test_tortuosity_density(self):
        self.assertEqual(
            tm.tortuosity_density([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            0,
            "Tortuosity Density should be zero")
        self.assertTrue(
            tm.tortuosity_density(
                [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 2, 1, 2, 3, 4, 5]) > 0,
            "Tortuosity Density should be greater than zero")

    def test_squared_curvature_tortuosity(self):
        self.assertEqual(
            tm.squared_curvature_tortuosity([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            0,
            "squared curvature tortuosity does not match")

    def test_smooth_tortuosity(self):
        self.assertEqual(tm.smooth_tortuosity_cubic(range(0, 11, 1), [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]), 0)
        
    def test_fractal_tortuosity_curve(self):
        val = tm.fractal_tortuosity_curve([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
        self.assertEqual(int(val), 1, "tortuosity of a line should be close to 1")
