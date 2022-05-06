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

"""tests for landamark module"""
from unittest import TestCase
from retipy.retina import Retina
from retipy import landmarks as l
import numpy as np
from numpy.testing import assert_array_equal


class TestLandmarks(TestCase):
    _resources = 'retipy/resources/images/'
    _image_file_name = 'test_landmarks.png'
    _image_path = _resources + _image_file_name
    _test_path = 'retipy/test/csv/'

    def setUp(self):
        self.image = Retina(None, self._image_path)

    def test_potential_landmarks(self):
        self.image.reshape_for_landmarks(2)
        self.image.threshold_image()
        self.image.skeletonization()
        skeleton = self.image.get_uint_image()
        landmarks, segments = l.potential_landmarks(skeleton, 3)
        result = np.genfromtxt(self._test_path + "potential_landmarks_test.csv", delimiter=',')

        assert_array_equal(result, landmarks, "landmark points does not match")

    def test_vessel_width(self):
        self.image.reshape_for_landmarks(2)
        self.image.threshold_image()
        threshold = self.image.get_uint_image()
        self.image.skeletonization()
        skeleton = self.image.get_uint_image()
        landmarks, segments = l.potential_landmarks(skeleton, 3)
        widths = l.vessel_width(threshold, landmarks)
        result = np.genfromtxt(self._test_path + "vessel_widths_test.csv", delimiter=',')

        assert_array_equal(result, widths, "Vessel widths does not match")

    def test_finding_landmark_vessels(self):
        self.image.reshape_for_landmarks(2)
        self.image.threshold_image()
        threshold = self.image.get_uint_image()
        self.image.skeletonization()
        skeleton = self.image.get_uint_image()
        self.image.bin_to_bgr()
        skeleton_rgb = self.image.get_uint_image()
        landmarks, segments = l.potential_landmarks(skeleton, 3)
        widths = l.vessel_width(threshold, landmarks)
        vessels = l.finding_landmark_vessels(widths, landmarks, skeleton, skeleton_rgb)
        result = np.genfromtxt(self._test_path + "finding_landmark_vessels_test.csv", delimiter=',')

        assert_array_equal(result, vessels[0], "Landmark vessels does not match")

    def test_vessel_number(self):
        self.image.reshape_for_landmarks(2)
        self.image.threshold_image()
        threshold = self.image.get_uint_image()
        self.image.skeletonization()
        skeleton = self.image.get_uint_image()
        self.image.bin_to_bgr()
        skeleton_rgb = self.image.get_uint_image()
        landmarks, segments = l.potential_landmarks(skeleton, 3)
        widths = l.vessel_width(threshold, landmarks)
        vessels = l.finding_landmark_vessels(widths, landmarks, skeleton, skeleton_rgb)
        marked_skeleton, final_landmarks = l.vessel_number(vessels, landmarks, skeleton_rgb)
        result_skeleton = np.genfromtxt(self._test_path + "vessel_number_skeleton_test.csv", delimiter=',')
        result_landmarks = np.genfromtxt(self._test_path + "vessel_number_filter_test.csv", delimiter=',')

        assert_array_equal(result_skeleton, marked_skeleton[20, :], "Vessel skeleton does not match")
        assert_array_equal(result_landmarks, final_landmarks, "Vessel final landmarks does not match")

    def test_principal_boxes(self):
        self.image.reshape_for_landmarks(2)
        self.image.threshold_image()
        threshold = self.image.get_uint_image()
        self.image.skeletonization()
        skeleton = self.image.get_uint_image()
        self.image.bin_to_bgr()
        skeleton_rgb = self.image.get_uint_image()
        landmarks, segments = l.potential_landmarks(skeleton, 3)
        widths = l.vessel_width(threshold, landmarks)
        vessels = l.finding_landmark_vessels(widths, landmarks, skeleton, skeleton_rgb)
        marked_skeleton, final_landmarks = l.vessel_number(vessels, landmarks, skeleton_rgb)
        bifurcations, crossings = l.principal_boxes(marked_skeleton, final_landmarks, 2)
        result = np.genfromtxt(self._test_path + "boxes_bifurcations_test.csv", delimiter=',')
        result2 = np.genfromtxt(self._test_path + "boxes_crossings_test.csv", delimiter=',')

        assert_array_equal(result, bifurcations[0], "Bifurcation points does not match")
        assert_array_equal(result2, crossings[0], "Crossing points does not match")

    def test_classification(self):
        bifurcations, crossings = l.classification(self.image.np_image, 2)
        result = np.genfromtxt(self._test_path + "boxes_bifurcations_test.csv", delimiter=',')
        result2 = np.genfromtxt(self._test_path + "boxes_crossings_test.csv", delimiter=',')

        assert_array_equal(result, bifurcations[0], "Bifurcation points does not match")
        assert_array_equal(result2, crossings[0], "Crossing points does not match")



