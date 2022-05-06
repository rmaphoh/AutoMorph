# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017  Maria Aguiar
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

"""tests for vessel classification module"""
from unittest import TestCase
from retipy.retina import Retina
from retipy import landmarks as l
from retipy import vessel_classification as vc
import numpy as np
from numpy.testing import assert_array_equal
import cv2
import h5py


class TestVesselClassification(TestCase):
    _resources = 'retipy/resources/images/'
    _image_file_name = 'manual.png'
    _image_path = _resources + _image_file_name
    _test_path = 'retipy/test/csv/'

    def setUp(self):
        self.manual = Retina(None, self._image_path)
        self.av = cv2.imread(self._resources+'av.png', 1)
        self.original = cv2.imread(self._resources+'original.tif', 1)

    def test_vessel_width(self):
        self.manual.threshold_image()
        threshold = self.manual.get_uint_image()
        self.manual.skeletonization()
        skeleton = self.manual.get_uint_image()
        landmarks, segmented = l.potential_landmarks(skeleton, 3)
        widths = vc._vessel_widths(segmented, threshold)

        result = np.genfromtxt(self._test_path + "vessels_width_test.csv", delimiter=',')
        assert_array_equal(result, widths[0:10], "Vessel widths does not match")

    def test_LBP(self):
        window = [[5, 8, 1], [5, 4, 1], [3, 7, 2]]
        self.assertEqual(46, vc._local_binary_pattern(window), "LBP wrong calculated, should return 162")

    def test_vectors(self):
        self.manual.threshold_image()
        threshold = self.manual.get_uint_image()
        self.manual.skeletonization()
        skeleton = self.manual.get_uint_image()
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        widths = vc._vessel_widths(skeleton, threshold)
        iv = vc._vector(widths[0], 6, self.original, L, gray, 0)

        result = np.genfromtxt(self._test_path + "vector_test.csv", delimiter=',')
        assert_array_equal(result, iv, "Feature vector does not match")

    def test_preparing_data_av(self):
        self.manual.threshold_image()
        threshold = self.manual.get_uint_image()
        self.manual.skeletonization()
        skeleton = self.manual.get_uint_image()
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        landmarks, segmented = l.potential_landmarks(skeleton, 3)
        widths = vc._vessel_widths(segmented, threshold)
        features = vc._preparing_data(widths, 6, self.original, self.av, L, gray)

        result = np.genfromtxt(self._test_path + "preparing_data_test.csv", delimiter=',')
        assert_array_equal(result, features[0:10], "Data does not match")

    def test_preparing_data_without_av(self):
        self.manual.threshold_image()
        threshold = self.manual.get_uint_image()
        self.manual.skeletonization()
        skeleton = self.manual.get_uint_image()
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        landmarks, segmented = l.potential_landmarks(skeleton, 3)
        widths = vc._vessel_widths(segmented, threshold)
        features = vc._preparing_data(widths, 6, self.original, None, L, gray)

        result = np.genfromtxt(self._test_path + "preparing_data_without_av_test.csv", delimiter=',')
        assert_array_equal(result, features[0:10], "Data does not match")

    def test_feature_vectors(self):
        vectors = vc._feature_vectors()
        h5f = h5py.File('retipy/retipy/model/vector_features_interpolation.h5', 'r')
        data = h5f['training'][:]
        h5f.close()

        assert_array_equal(data[0:10], vectors[0:10], "Vectors does not match")

    def test_loading_model(self):
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        result = np.genfromtxt(self._test_path + "loading_model_segments_test.csv", delimiter=',')
        result2 = np.genfromtxt(self._test_path + "loading_model_predictions_test.csv", delimiter=',')

        assert_array_equal(result, segments[:, 20], "Segmented skeleton image does not match")
        assert_array_equal(result2, predictions[:, 20], "Neural Network predictions does not match")

    def test_validating_model(self):
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        acc, rgb, network, original = vc._validating_model(features, segments, self.original, predictions, 38, 1)
        self.assertEqual(76.18243243243244, acc,  "Wrong validation, should return 81.1214953271028")

    def test_validating_model_without_av(self):
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        acc, rgb, network, original = vc._validating_model(features, segments, self.original, predictions, 38, 0)
        self.assertEqual(-1, acc,  "Wrong validation, should return -1")

    def test_homogenize(self):
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        acc, rgb, network, original = vc._validating_model(features, segments, self.original, predictions, 38, 1)
        connected_components = cv2.connectedComponentsWithStats(segments.astype(np.uint8), 4, cv2.CV_32S)
        final_img, rgb_img = vc._homogenize(connected_components, network, rgb, original)

        result = np.genfromtxt(self._test_path + "homogenize_test.csv", delimiter=',')
        assert_array_equal(result, rgb_img[:, 20], "Homogenized image does not match")

    def test_box_labels(self):
        bifurcations, crossings = l.classification(self.manual.np_image, 0)
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        connected_components = cv2.connectedComponentsWithStats(segments.astype(np.uint8), 4, cv2.CV_32S)
        connected_vessels = vc._box_labels(bifurcations, connected_components)

        result = np.genfromtxt(self._test_path + "box_labels_test.csv", delimiter=',')
        assert_array_equal(result, connected_vessels, "Box labels does not match")

    def test_average(self):
        self.assertEqual(7, vc._average([[1, 3, 4], [2, 5, 4], [3, 2, 3]]),  "Average width wrong calculated, should return 7")

    def test_average_width(self):
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        connected_components = cv2.connectedComponentsWithStats(segments.astype(np.uint8), 4, cv2.CV_32S)
        bifurcations, crossings = l.classification(self.manual.np_image, 0)
        connected_vessels = vc._box_labels(bifurcations, connected_components)
        acc, rgb, network, original = vc._validating_model(features, segments, self.original, predictions, 38, 1)
        final_img, rgb_img = vc._homogenize(connected_components, network, rgb, original)
        widths_colors = vc._average_width(connected_components, connected_vessels[0], thr, rgb_img)
        wc = [widths_colors[2]]
        wc.extend(widths_colors[3])

        result = np.genfromtxt(self._test_path + "average_width_test.csv", delimiter=',')
        assert_array_equal(result, wc, "Width and color do not match")

    def test_normalize_indexes(self):
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        connected_components = cv2.connectedComponentsWithStats(segments.astype(np.uint8), 4, cv2.CV_32S)
        normal = vc._normalize_indexes(connected_components, 7)

        result = np.genfromtxt(self._test_path + "normalize_indexes_test.csv", delimiter=',')
        assert_array_equal(result, normal, "Width and color do not match")

    def test_coloring(self):
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        connected_components = cv2.connectedComponentsWithStats(segments.astype(np.uint8), 4, cv2.CV_32S)
        bifurcations, crossings = l.classification(self.manual.np_image, 0)
        connected_vessels = vc._box_labels(bifurcations, connected_components)
        acc, rgb, network, original = vc._validating_model(features, segments, self.original, predictions, 38, 1)
        final_img, rgb_img = vc._homogenize(connected_components, network, rgb, original)
        rgb = vc._coloring(connected_components, connected_vessels[0], [0, 0, 255], rgb_img)

        assert_array_equal([0, 0, 255], rgb, "Coloring does not match")

    def test_postprocessing(self):
        bifurcations, crossings = l.classification(self.manual.np_image, 0)
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        acc, rgb, network, original = vc._validating_model(features, segments, self.original, predictions, 38, 1)
        connected_components = cv2.connectedComponentsWithStats(segments.astype(np.uint8), 4, cv2.CV_32S)
        final_img, rgb_img = vc._homogenize(connected_components, network, rgb, original)
        post_img = vc._postprocessing(connected_components, thr, bifurcations, rgb_img)

        result = np.genfromtxt(self._test_path + "postprocessing_test.csv", delimiter=',')
        assert_array_equal(result, post_img[:, 20], "Post image does not match")

    def test_accuracy(self):
        bifurcations, crossings = l.classification(self.manual.np_image, 0)
        features, segments, thr, predictions = vc._loading_model(self.original, self.manual.np_image, self.av, 38)
        acc, rgb, network, original = vc._validating_model(features, segments, self.original, predictions, 38, 1)
        connected_components = cv2.connectedComponentsWithStats(segments.astype(np.uint8), 4, cv2.CV_32S)
        final_img, rgb_img = vc._homogenize(connected_components, network, rgb, original)

        post_img = vc._postprocessing(connected_components, thr, bifurcations, rgb_img)
        acc = vc._accuracy(post_img, segments, self.av)

        assert_array_equal([0.8447412353923205, 0.7686274509803922, 0.9011627906976745], acc, "Accuracy does not match")

    def test_classification(self):
        post_img = vc.classification(self.original, self.manual.np_image)

        result = np.genfromtxt(self._test_path + "classification_test.csv", delimiter=',')
        assert_array_equal(result, post_img[:, 20], "Classificated image does not match")
