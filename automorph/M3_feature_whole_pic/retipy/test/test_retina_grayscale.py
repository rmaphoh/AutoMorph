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
import cv2
from retipy import retina_grayscale

_resources = 'retipy/test/resources/images/'
_image_file_name = 'test_image.pgm'
_image_file_name_2 = '01_h.jpg'
_image_path = _resources + _image_file_name
_image_path_2 = _resources + _image_file_name_2
_shadow_correction_path = _resources + 'shadowCorrection.pgm'
_homogenize_path = _resources + 'homogenization.pgm'
_normal_vessels_segmentation_path = _resources + 'normalVesselsSegmentation.pgm'
_tiny_vessels_segmentation_path = _resources + 'tinyVesselsSegmentation.pgm'
_double_vessels_segmentation_path = _resources + 'finalVesselsSegmentation.pgm'
_postprocessing_path = _resources + 'postProcessing.pgm'
_manual_result_path = _resources + 'result.pgm'


class TestRetinaGrayscale(TestCase):
    """Test class for Retina class"""

    def setUp(self):
        self.image = retina_grayscale.Retina_grayscale(None, _image_path, 1)

    def test_constructor_invalid_path(self):
        """Test the retina constructor when the given path is invalid"""
        self.assertRaises(Exception, retina_grayscale.Retina_grayscale, None, _resources)

    def test_constructor_existing_image(self):
        """Test the constructor with an existing image"""
        image = io.imread(_image_path)
        none_constructor_image = retina_grayscale.Retina_grayscale(image, _image_file_name, 1)

    def test_constructor_image_type_2(self):
        """Test the constructor with an existing image"""
        image = io.imread(_image_path)
        none_constructor_image = retina_grayscale.Retina_grayscale(image, _image_file_name, 2)

    def test_constructor_image_no_type(self):
        """Test the constructor with an existing image"""
        image = io.imread(_image_path)
        none_constructor_image = retina_grayscale.Retina_grayscale(image, _image_file_name)

    def test_constructor_image_no_type_2(self):
        """Test the constructor with an existing image"""
        image = io.imread(_image_path_2)
        none_constructor_image = retina_grayscale.Retina_grayscale(image, _image_file_name)

    def test_restore_mask(self):
        self.image.np_image[self.image.mask == 0] = 5
        self.image.restore_mask()
        original_image = retina_grayscale.Retina_grayscale(None, _image_path, 1)
        original_image.np_image[original_image.mask == 0] = 5
        original_image.restore_mask()
        assert_array_equal(self.image.np_image, original_image.np_image)

    def test_equalize_histogram(self):
        self.image.equalize_histogram()
        original_image = retina_grayscale.Retina_grayscale(None, _image_path, 1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        original_image.np_image = clahe.apply(original_image.np_image)
        original_image.restore_mask()
        assert_array_equal(self.image.np_image, original_image.np_image)

    def test_opening(self):
        self.image.opening(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path, 1)
        assert_array_equal(self.image.np_image,
                           ndimage.grey_opening(original_image.np_image, size=(3, 3)))

    def test_closing(self):
        self.image.closing(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path, 1)
        assert_array_equal(self.image.np_image,
                           ndimage.grey_closing(original_image.np_image, size=(3, 3)))

    def test_top_hat(self):
        self.image.top_hat(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path, 1)
        assert_array_equal(self.image.np_image,
                           cv2.morphologyEx(original_image.np_image, cv2.MORPH_TOPHAT,
                                            cv2.getStructuringElement(cv2.MORPH_RECT, (
                                                3, 3))))

    def test_mean_filter(self):
        self.image.mean_filter(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path, 1)
        assert_array_equal(self.image.np_image,
                           cv2.blur(original_image.np_image, (3, 3)))

    def test_gaussian_filter(self):
        self.image.gaussian_filter(17, 1.82)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path, 1)
        assert_array_equal(self.image.np_image,
                           cv2.GaussianBlur(original_image.np_image, (17, 17), 1.82))

    def test_median_filter(self):
        self.image.median_filter(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path, 1)
        assert_array_equal(self.image.np_image,
                           cv2.medianBlur(original_image.np_image.astype(np.uint8), 3))

    def test_shadow_correction(self):
        self.image.shadow_correction()

    def test_homogenize(self):
        self.image.shadow_correction()
        self.image.homogenize()

    def test_tiny_vessels_segmentation(self):
        tiny_segmentation = self.image.tiny_vessels_segmentation()

    def test_normal_vessels_segmentation(self):
        normal_segmentation = self.image.normal_vessels_segmentation()

    def test_post_processing(self):
        img = self.image.normal_vessels_segmentation()
        result = self.image.post_processing(img)

    def test_double_vessels_segmentation(self):
        double_segmentation = self.image.double_segmentation()
        other_segmentation = retina_grayscale.Retina_grayscale(None, _image_path,
                                                               1).double_segmentation()
        assert_array_equal(double_segmentation, other_segmentation)

    def test_calculate_roc(self):
        double_segmentation = self.image.normal_vessels_segmentation()
        original_image = retina_grayscale.Retina_grayscale(None, _manual_result_path, 1)
        self.image.calculate_roc(double_segmentation / 255, original_image.np_image / 255)
        np.testing.assert_allclose(self.image.roc, [[425., 304., 423., 299., 1451.]], 1e-2)
