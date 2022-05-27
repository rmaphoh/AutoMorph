# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Luis Felipe Castaño
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

"""retina module to handle basic image processing on retinal images"""

import base64
from copy import copy
import numpy as np
import cv2
from scipy import ndimage
from skimage import io
from matplotlib import pyplot as plt
from os import path
from PIL import Image
from io import BytesIO

class Retina_grayscale(object):
    """
    Retina_grayscale class that internally contains a matrix with the green channel image data for a retinal image, it
    constructor expects a path to the image

    :param image: a numpy array with the image data
    :param image_path: path to an image to be open
    :param image_type: This value represent the image resolution. When this value is zero, the algorithm is set automatically
    """
    @staticmethod
    def _open_image(img_path):
        return io.imread(img_path)

    @staticmethod
    def get_base64_image(image: np.ndarray):
        temp_image = Image.fromarray(image.astype('uint8'), 'L')
        buffer = BytesIO()
        temp_image.save(buffer, format="png")
        return str(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    def __init__(self, image: np.ndarray, image_path: str, image_type: int=0):
        if image is None:
            self.np_image = self._open_image(image_path)
            _, file = path.split(image_path)
            self._file_name = file
        else:
            self.np_image = image
            self._file_name = image_path

        self.np_image = self.np_image[:, :, 1]
        """if(self.np_image.shape[0] == 3328):
            self.np_image[3270:3328, :] = 0"""
        self.old_image = None
        self.shape = self.np_image.shape
        self.original_image = self.np_image
        self.segmented = False
        self.segmented_image = np.zeros((self.shape))
        self.roc = np.zeros((1,5)).astype(np.float)

        if image_type == 0:
            if self.shape[0] <= 1020 and self.shape[1] <= 1020:
                image_type = 2
            else:
                image_type = 1

        if image_type == 1:
            self.mask = self.np_image <= 5
            self.kernel_mean_filter = 11
            self.kernel_gaussian_filter = 33
            self.kernel_median_filter = 111
            self.kernel_opening = 13 #9
            self.normal_vessels_segmentation_min_value = 5000
            self.main_adaptative_method = cv2.ADAPTIVE_THRESH_MEAN_C
            self.tiny_vessels_segmentation_min_value = 100
            self.postprocesing_segmentation_min_value = 70
            self.maximum_radius_to_fill = 20
            self.tiny_vessels_threshold = 13 #11
            self.kernel_erode = 3
            self.kernel_dilate = 3
            self.smoothing_curves_iterations = 6
            self.smoothing_curves_kernel = 5
        if image_type == 2:
            self.mask = self.np_image <= 30
            self.kernel_mean_filter = 3
            self.kernel_gaussian_filter = 9
            self.kernel_median_filter = 41
            self.kernel_opening = 3 #9
            self.normal_vessels_segmentation_min_value = 50
            self.main_adaptative_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            self.tiny_vessels_segmentation_min_value = 150
            self.postprocesing_segmentation_min_value = 17
            self.maximum_radius_to_fill = 5
            self.tiny_vessels_threshold = 11#11
            self.kernel_erode = 1
            self.kernel_dilate = 1
            self.smoothing_curves_iterations = 2
            self.smoothing_curves_kernel = 3

        self.mask = abs(1 - self.mask)

##################################################################################################
# Image Processing functions

    def restore_mask(self):
        """Restores the mask when it has been affected by the application of a filter"""
        mask = self.mask == 0
        self.np_image[mask] = 0

    def equalize_histogram(self):
        """Applies contrast limited adaptive histogram equalization algorithm (CLAHE)"""
        self._copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        self.np_image = clahe.apply(self.np_image)
        self.restore_mask()

    def opening(self, size_structure):
        """
        dilates and erodes the stored image, by default the structure is a cross
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = ndimage.grey_opening(self.np_image, size=(size_structure, size_structure))

    def closing(self, size_structure):
        """
        erodes and dilates the stored image, by default the structure is a cross
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = ndimage.grey_closing(self.np_image, size=(size_structure, size_structure))

    def top_hat(self, size_structuring_element):
        """
        Applies Top-hat filter
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = cv2.morphologyEx(self.np_image, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (size_structuring_element, size_structuring_element)))

    def mean_filter(self, structure):
        """
        Applies mean filter
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = cv2.blur(self.np_image,(structure,structure))#signal.medfilt(self.np_image, structure)

    def gaussian_filter(self, structure, sigma):
        """
        Applies Gaussian filter
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = cv2.GaussianBlur(self.np_image, (structure, structure), sigma)

    def median_filter(self, structure):
        """
        Applies median filter
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = cv2.medianBlur(self.np_image.astype(np.uint8), structure)#ndimage.median_filter(self.np_image, size=(structure, structure))

    def shadow_correction(self):
        """Applies the following filters: mean filter with a 3x3 kernel, Gaussian filter with a kernel of 9x9
        and sigma of 1.82, and a median filter with a 60x60 kernel. The resulting image is subtracted with the
        original image and finally the values obtained from the subtraction are moved to the 256 possible grayscale values"""

        self._copy()
        minuendo = np.copy(self.np_image)
        self.mean_filter(self.kernel_mean_filter)
        self.gaussian_filter(self.kernel_gaussian_filter, 1.82)
        self.mean_value = np.mean(self.np_image)
        self.np_image[self.mask] = self.mean_value
        self.median_filter(self.kernel_median_filter)#Ver si es posible aumentarlo en otro pc
        self.np_image = minuendo - self.np_image.astype(np.float)
        min = self.np_image.min()
        self.np_image = self.np_image - min
        max = self.np_image.max()
        escala = float(255) / (max)
        for row in range(0, self.shape[0]):
            for col in range(0, self.shape[1]):
                self.np_image[row, col] = int(self.np_image[row, col] * escala)
        self.restore_mask()

    def homogenize(self):
        """Moves all the values resulting from the correction of the shadows to the possible 255 values"""
        self._copy()
        g_input_max = self.np_image.max()
        aux = np.zeros(self.shape)
        for row in range(0, self.shape[0]):
            for col in range(0, self.shape[1]):
                g = self.np_image[row, col] + 180 - g_input_max
                if (g < 0):
                    aux[row, col] = 0
                elif (g > 255): # pragma: no cover
                    aux[row, col] = 255
                else:
                    aux[row, col] = g
        self.np_image = aux
        self.IH = np.copy(aux)

    def normal_vessels_segmentation(self):
        self.shadow_correction()
        self.homogenize()
        IH = cv2.GaussianBlur(self.IH, (3, 3), 1.72).astype(np.uint8)
        ret, normal_vessels_segmentation = cv2.threshold(IH, 0, 255, cv2.THRESH_OTSU)
        npaContours, hierarchy = cv2.findContours(normal_vessels_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for npaContour in npaContours:
            if (cv2.contourArea(npaContour) < self.normal_vessels_segmentation_min_value):  # 100
                cv2.drawContours(normal_vessels_segmentation, [npaContour], -1, (255, 255, 255), -1)
        return abs(255 - normal_vessels_segmentation)


    def tiny_vessels_segmentation(self):
        self.equalize_histogram()
        self.opening(self.kernel_opening)
        self.shadow_correction()
        self.homogenize()
        IH = cv2.GaussianBlur(self.IH, (3, 3), 1.72).astype(np.uint8)

        tiny_vessels_segmentation = cv2.adaptiveThreshold(IH, 255, self.main_adaptative_method, cv2.THRESH_BINARY, self.tiny_vessels_threshold, 2)#13

        tiny_vessels_segmentation = abs(255 - tiny_vessels_segmentation)

        kernel = np.ones((self.kernel_erode, self.kernel_erode), np.uint8)
        tiny_vessels_segmentation = cv2.erode(tiny_vessels_segmentation, kernel, iterations=1)

        npaContours, hierarchy = cv2.findContours(tiny_vessels_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for npaContour in npaContours:
            if (cv2.contourArea(npaContour) < self.tiny_vessels_segmentation_min_value):  # pragma: no cover
                cv2.drawContours(tiny_vessels_segmentation, [npaContour], -1, (0, 0, 0), -1) # 250

        kernel = np.ones((self.kernel_dilate, self.kernel_dilate), np.uint8)
        tiny_vessels_segmentation = cv2.dilate(tiny_vessels_segmentation, kernel, iterations=1)
        return tiny_vessels_segmentation

    def post_processing(self, final_vessels_segmentation):
        kernel = np.ones((self.kernel_dilate, self.kernel_dilate), np.uint8)
        final_vessels_segmentation = cv2.dilate(final_vessels_segmentation, kernel, iterations=1)
        final_vessels_segmentation = abs(255 - final_vessels_segmentation)

        npaContours, hierarchy = cv2.findContours(final_vessels_segmentation.astype(np.uint8), cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        for npaContour in npaContours:
            (x, y), radius = cv2.minEnclosingCircle(npaContour)
            if (radius < self.postprocesing_segmentation_min_value):
                center = (int(x), int(y))
                cv2.drawContours(final_vessels_segmentation, [npaContour], -1, (255, 255, 255), -1)

        final_vessels_segmentation = abs(255 - final_vessels_segmentation)

        npaContours, hierarchy = cv2.findContours(final_vessels_segmentation.astype(np.uint8), cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        for npaContour in npaContours:
            (x, y), radius = cv2.minEnclosingCircle(npaContour)
            if (radius < self.maximum_radius_to_fill):
                center = (int(x), int(y))
                # cv2.circle(final_vessels_segmentation, center, int(radius), (0, 0, 0), 2)
                cv2.drawContours(final_vessels_segmentation, [npaContour], -1, (255, 255, 255), -1)
        cv2.pyrUp(final_vessels_segmentation, final_vessels_segmentation)
        for i in range(0, self.smoothing_curves_iterations):
            final_vessels_segmentation = cv2.medianBlur(final_vessels_segmentation.astype(np.uint8), self.smoothing_curves_kernel)

        cv2.pyrDown(final_vessels_segmentation, final_vessels_segmentation)

        kernel = np.ones((self.kernel_erode, self.kernel_erode), np.uint8)
        final_vessels_segmentation = cv2.erode(final_vessels_segmentation, kernel, iterations=1)
        return final_vessels_segmentation

    def double_segmentation(self):
        normal_vessels_segmentation = self.normal_vessels_segmentation()
        self.np_image = self.original_image
        tiny_vessels_segmentation = self.tiny_vessels_segmentation()
        final_vessels_segmentation = tiny_vessels_segmentation.astype(np.float) + normal_vessels_segmentation.astype(np.float)
        k = final_vessels_segmentation >= 1
        final_vessels_segmentation[k] = 255
        k2 = final_vessels_segmentation <= 0
        final_vessels_segmentation[k2] = 0

        final_vessels_segmentation = self.post_processing(final_vessels_segmentation)
        return self.get_base64_image(final_vessels_segmentation)

    def calculate_roc(self, image, result):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for i in range(0, result.shape[0]):
            for j in range(0, result.shape[1]):
                if (self.mask[i, j] == 1):
                    if (image[i, j] == 1 and result[i, j] == 1):
                        true_positive += 1
                    elif (image[i, j] == 0 and result[i, j] == 0):
                        true_negative += 1
                    elif (image[i, j] == 1 and result[i, j] == 0):
                        false_positive += 1
                    elif (image[i, j] == 0 and result[i, j] == 1):
                        false_negative += 1
        print(true_positive, true_negative, false_positive, false_negative)
        self.roc[0, :] = [true_positive, true_negative, false_positive, false_negative,
                          (true_positive + true_negative + false_positive + false_negative)]

    ##################################################################################################
# I/O functions

    def _copy(self):
        self.old_image = copy(self.np_image)

    def view(self):  # pragma: no cover
        """show a window with the internal image"""
        io.imshow(self.np_image)
        plt.show()


