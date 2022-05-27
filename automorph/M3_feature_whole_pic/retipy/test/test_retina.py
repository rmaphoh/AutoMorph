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

"""tests for retina module"""

import os
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from scipy import ndimage
from skimage import color, filters, io
from skimage.morphology import skeletonize

from retipy import retina

_resources = 'retipy/resources/images/'
_image_file_name = 'img01.png'
_image_path = _resources + _image_file_name


class TestRetina(TestCase):
    """Test class for Retina class"""

    def setUp(self):
        self.image = retina.Retina(None, _image_path)

    def tearDown(self):
        if os.path.isfile("./out_" + _image_file_name):
            os.unlink("./out_" + _image_file_name)

    def test_constructor_invalid_path(self):
        """Test the retina constructor when the given path is invalid"""
        self.assertRaises(Exception, retina.Retina, None, _resources)

    def test_constructor_existing_image(self):
        """Test the constructor with an existing image"""
        image = retina.Retina(None, _image_path)
        none_constructor_image = retina.Retina(image.np_image, _image_file_name)

        assert_array_equal(image.np_image, none_constructor_image.np_image, "created images should be the same")

    def test_segmented(self):
        """Test default value for segmented property"""
        self.assertEqual(
            False, self.image.segmented, "segmented should be false by default")
        self.image.segmented = True
        self.assertEqual(
            True, self.image.segmented, "segmented should be true")

    def test_threshold_image(self):
        self.image.threshold_image()
        original_image = color.rgb2gray(io.imread(_image_path))
        output = original_image > filters.threshold_mean(original_image)

        assert_array_equal(self.image.np_image, output, "segmented image does not match")

    def test_apply_thinning(self):
        retina_image = retina.Retina(np.zeros((64, 64), np.uint8), _image_file_name)
        retina_image.np_image[10:17, 10:13] = 1
        retina_image.apply_thinning()
        output = [0, 1, 1, 1, 1, 0]
        assert_array_equal(retina_image.np_image[10:16, 11], output, "expected a line")

    def test_save_image(self):
        self.image.save_image(".")
        self.assertTrue(os.path.isfile("./out_" + _image_file_name))

    def test_undo(self):
        self.image.detect_edges_canny()
        original_image = retina.Retina(None, _image_path)
        self.assertRaises(
            AssertionError,
            assert_array_equal,
            self.image.np_image, original_image.np_image, "images should be different")
        self.image.undo()
        assert_array_equal(self.image.np_image, original_image.np_image, "image should be the same")

    def test_erode(self):
        self.image.threshold_image()
        self.image.erode(1)
        original_image = retina.Retina(None, _image_path)
        original_image.threshold_image()
        assert_array_equal(
            self.image.np_image, ndimage.binary_erosion(original_image.np_image, iterations=1))

    def test_dilate(self):
        self.image.threshold_image()
        self.image.dilate(1)
        original_image = retina.Retina(None, _image_path)
        original_image.threshold_image()
        assert_array_equal(
            self.image.np_image, ndimage.binary_dilation(original_image.np_image, iterations=1))

    def test_compare_with(self):
        self.image.threshold_image()
        original_image = retina.Retina(None, _image_path)
        assert_array_equal(
            self.image.compare_with(original_image).np_image,
            self.image.np_image - original_image.np_image,
            "image does not match")

    def test_reshape_image(self):
        self.image.reshape_square()
        self.assertEqual(self.image.shape[0], self.image.shape[1], "dimension should be the same when reshaping")

    def test_get_window_sizes(self):
        windows = self.image.get_window_sizes()
        assert_array_equal(windows, [], "window array does not match")
        self.image.reshape_square()
        windows = self.image.get_window_sizes()
        assert_array_equal(windows, [292,146,73], "window array does not match")

    def test_reshape_by_window(self):
        print(self.image.shape)
        self.image.reshape_by_window(32)
        print(self.image.shape)
        self.assertEqual(self.image.shape[0] % 32, 0, "modulo should be zero after reshape")
        self.assertEqual(self.image.shape[1] % 32, 0, "modulo should be zero after reshape")

    def test_skeletonization(self):
        self.image.threshold_image()
        self.image.skeletonization()
        original_image = retina.Retina(None, _image_path)
        original_image.threshold_image()
        assert_array_equal(self.image.np_image, skeletonize(original_image.np_image))

    def test_bin_to_bgr(self):
        h, w = self.image.shape
        image_bgr = np.zeros((h, w, 3))
        image_bgr[:, :, 0] = self.image.np_image
        image_bgr[:, :, 1] = self.image.np_image
        image_bgr[:, :, 2] = self.image.np_image
        self.image.bin_to_bgr()
        assert_array_equal(image_bgr, self.image.np_image)

    def test_uint_image(self):
        image = self.image.np_image.astype(np.uint8) * 255
        assert_array_equal(image, self.image.get_uint_image())

    def test_reshape_to_landmarks(self):
        self.image.reshape_for_landmarks(5)
        new_image = retina.Retina(None, _image_path)
        new_image.np_image = np.pad(new_image.np_image, pad_width=5, mode='constant', constant_values=0)
        assert_array_equal(new_image.np_image, self.image.np_image)


class TestWindow(TestCase):

    _image_size = 64

    def setUp(self):
        self._retina_image = retina.Retina(
            np.zeros((self._image_size, self._image_size), np.uint8), _image_file_name)

    def test_create_windows(self):
        # test with an empty image
        self.assertRaises(ValueError, retina.Window, self._retina_image, 8)

        # test with a full data image
        self._retina_image.np_image[:, :] = 1
        windows = retina.Window(self._retina_image, 8)
        self.assertEqual(windows.windows.shape[0], self._image_size, "expected 64 windows")

        # test with an image half filled with data
        self._retina_image.np_image[:, 0:int(self._image_size/2)] = 0
        windows = retina.Window(self._retina_image, 8)
        self.assertEqual(windows.windows.shape[0], self._image_size/2, "expected 32 windows")

    def test_create_windows_error_dimension(self):
        self.assertRaises(ValueError, retina.Window, self._retina_image, 7)

    def test_create_windows_combined(self):
        windows = retina.Window(self._retina_image, 8, "combined", 0)

        # combined should create (width/(dimension/2) - 1) * (height/(dimension/2) -1)
        # here is (64/4 -1) * (64/4 -1) = 225
        self.assertEqual(windows.windows.shape[0], 225, "there should be 225 windows created")

        # fail with no window created
        self.assertRaises(ValueError, retina.Window, self._retina_image, 8, "combined")

    def test_create_windows_combined_error_dimension(self):
        new_image = retina.Retina(np.zeros((66, 66), np.uint8), _image_file_name)
        self.assertRaises(ValueError, retina.Window, new_image, 33, "combined", 0)

    def test_vessel_extractor(self):
        self._retina_image.np_image[10, 10:20] = 1
        self._retina_image.np_image[11, 20] = 1
        self._retina_image.np_image[9, 20] = 1
        self._retina_image.np_image[11, 21] = 1
        self._retina_image.np_image[9, 21] = 1
        vessels = retina.detect_vessel_border(self._retina_image)

        self.assertEqual(len(vessels), 1, "only one vessel should've been extracted")
        self.assertEqual(len(vessels[0][0]), 3, "vessel should have 3 pixels")

    def test_save_window(self):
        self._retina_image.np_image[:, :] = 1
        window = retina.Window(self._retina_image, 8, min_pixels=0)
        window.save_window(1, "./")
        self.assertTrue(os.path.isfile(window._window_filename(1)), "file not found")
        os.unlink(window._window_filename(1))

    def test_save_window_wrong_id(self):
        self._retina_image.np_image[:, :] = 1
        window = retina.Window(self._retina_image, 8, min_pixels=0)
        self.assertRaises(ValueError, window.save_window, 80, "./")

    def test_mode(self):
        window = retina.Window(self._retina_image, 8, min_pixels=0)
        assert_array_equal(window.shape, [64, 1, 8, 8], "window shape is incorrect")

        window.mode = window.mode_tensorflow
        assert_array_equal(window.shape, [64, 8, 8, 1], "window shape is incorrect")

        window.mode = window.mode_pytorch
        assert_array_equal(window.shape, [64, 1, 8, 8], "window shape is incorrect")

    def test_tags(self):
        window = retina.Window(self._retina_image, 8, min_pixels=0)
        tags = np.zeros([window.shape[0], 4])
        window.tags = tags
        assert_array_equal(tags, window.tags, "tags does not match")

    def test_tags_wrong_input(self):
        window = retina.Window(self._retina_image, 8, min_pixels=0)
        with self.assertRaises(ValueError):
            tags = np.zeros([1, 4])
            window.tags = tags

    def test_create_tag_image(self):
        image = retina.Window._create_tag_image(6, 6, [0, 50, 100, 150, 200, 255])
        # retina.Retina(image, "test").view()

    def test_set_tag_layer(self):
        image = retina.Retina(None, _image_path)
        image.reshape_by_window(56)
        window = retina.Window(image, 56, min_pixels=10)
        tags = np.full([window.shape[0], 4], 100)
        tags[:, 3] = 50
        window.tags = tags
        window.set_tag_layer()

    def test_set_tag_layer_tags_not_set(self):
        image = retina.Retina(None, _image_path)
        image.reshape_by_window(56)
        window = retina.Window(image, 56, min_pixels=10)
        with self.assertRaises(ValueError):
            window.set_tag_layer()

    def test_set_tag_layer_mode_tensorflow(self):
        image = retina.Retina(None, _image_path)
        image.reshape_by_window(56)
        window = retina.Window(image, 56, min_pixels=10)
        window.mode = window.mode_tensorflow
        tags = np.full([window.shape[0], 4], 100)
        tags[:, 3] = 50
        window.tags = tags
        window.set_tag_layer()

    def test_iterator(self):
        image = retina.Retina(None, _image_path)
        image.reshape_by_window(56)
        windows = retina.Window(image, 56, min_pixels=10)
        size = windows.windows.shape[0]
        iterated_size = 0
        for _ in windows:
            iterated_size += 1
        self.assertEqual(size, iterated_size, "iterated structure size does not match")

        for window in windows:
            assert_array_equal(window, windows.windows[0])
            break
