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

"""tests for configuration module"""

import configparser
import os
from unittest import TestCase

from retipy import configuration


class TestConfiguration(TestCase):
    """Test class for Configuration class of configuration module"""

    _image_directory = 'some/directory'
    _window_size = 2
    _pixels_per_window = 4
    _sampling_size = 6
    _r2_threshold = 0.94
    _output_folder = 'test'
    _config_file = 'test.config'

    def setUp(self):
        test_configuration = configparser.ConfigParser()
        test_configuration[configuration.PROPERTY_DEFAULT_CATEGORY] = {
            configuration.PROPERTY_IMAGE_DIRECTORY: self._image_directory,
            configuration.PROPERTY_WINDOW_SIZE: str(self._window_size),
            configuration.PROPERTY_PIXELS_PER_WINDOW: str(self._pixels_per_window),
            configuration.PROPERTY_SAMPLING_SIZE: str(self._sampling_size),
            configuration.PROPERTY_R2_THRESHOLD: str(self._r2_threshold),
            configuration.PROPERTY_OUTPUT_FOLDER:self._output_folder
        }
        with open(self._config_file, 'w') as configfile:
            test_configuration.write(configfile)

    def tearDown(self):
        os.unlink(self._config_file)

    def test_constructor(self):
        """test the constructor in a positive scenario"""
        config = configuration.Configuration(self._config_file)
        self.assertEqual(config.image_directory, self._image_directory, "wrong image directory")
        self.assertEqual(config.window_size, self._window_size)

    def test_constructor_no_default_cat(self):
        """test the constructor when there is no General category"""
        test_configuration = configparser.ConfigParser()
        with open(self._config_file, 'w') as configfile:
            test_configuration.write(configfile)
        self.assertRaises(
            configuration.ConfigurationException, configuration.Configuration, self._config_file)

    def test_constructor_error_img_dir(self):
        """test the constructor when the image directory is not defined"""
        test_configuration = configparser.ConfigParser()
        test_configuration['General'] = {
            configuration.PROPERTY_WINDOW_SIZE: str(self._window_size)
        }
        with open(self._config_file, 'w') as configfile:
            test_configuration.write(configfile)
        self.assertRaises(
            configuration.ConfigurationException, configuration.Configuration, self._config_file)

    def test_constructor_error_window(self):
        """test the constructor when the window property is not defined"""
        test_configuration = configparser.ConfigParser()
        test_configuration['General'] = {
            configuration.PROPERTY_IMAGE_DIRECTORY: self._image_directory
        }
        with open(self._config_file, 'w') as configfile:
            test_configuration.write(configfile)
        self.assertRaises(
            configuration.ConfigurationException, configuration.Configuration, self._config_file)

    def test_constructor_error_pixels(self):
        """test the constructor when the pixels per window property is not configured"""

        test_configuration = configparser.ConfigParser()
        test_configuration[configuration.PROPERTY_DEFAULT_CATEGORY] = {
            configuration.PROPERTY_IMAGE_DIRECTORY: self._image_directory,
            configuration.PROPERTY_WINDOW_SIZE: str(self._window_size),
            configuration.PROPERTY_SAMPLING_SIZE: str(self._sampling_size),
            configuration.PROPERTY_R2_THRESHOLD: str(self._r2_threshold),
            configuration.PROPERTY_OUTPUT_FOLDER:self._output_folder
        }
        with open(self._config_file, 'w') as configfile:
            test_configuration.write(configfile)
        self.assertRaises(
            configuration.ConfigurationException, configuration.Configuration, self._config_file)

    def test_constructor_error_sampling_size(self):
        test_configuration = configparser.ConfigParser()
        test_configuration[configuration.PROPERTY_DEFAULT_CATEGORY] = {
            configuration.PROPERTY_IMAGE_DIRECTORY: self._image_directory,
            configuration.PROPERTY_WINDOW_SIZE: str(self._window_size),
            configuration.PROPERTY_PIXELS_PER_WINDOW: str(self._pixels_per_window),
            configuration.PROPERTY_R2_THRESHOLD: str(self._r2_threshold),
            configuration.PROPERTY_OUTPUT_FOLDER:self._output_folder
        }
        with open(self._config_file, 'w') as configfile:
            test_configuration.write(configfile)
        self.assertRaises(
            configuration.ConfigurationException, configuration.Configuration, self._config_file)

    def test_constructor_error_r2_threshold(self):
        test_configuration = configparser.ConfigParser()
        test_configuration[configuration.PROPERTY_DEFAULT_CATEGORY] = {
            configuration.PROPERTY_IMAGE_DIRECTORY: self._image_directory,
            configuration.PROPERTY_WINDOW_SIZE: str(self._window_size),
            configuration.PROPERTY_PIXELS_PER_WINDOW: str(self._pixels_per_window),
            configuration.PROPERTY_SAMPLING_SIZE: str(self._sampling_size),
            configuration.PROPERTY_OUTPUT_FOLDER:self._output_folder
        }
        with open(self._config_file, 'w') as configfile:
            test_configuration.write(configfile)
        self.assertRaises(
            configuration.ConfigurationException, configuration.Configuration, self._config_file)

    def test_constructor_error_output_folder(self):
        test_configuration = configparser.ConfigParser()
        test_configuration[configuration.PROPERTY_DEFAULT_CATEGORY] = {
            configuration.PROPERTY_IMAGE_DIRECTORY: self._image_directory,
            configuration.PROPERTY_WINDOW_SIZE: str(self._window_size),
            configuration.PROPERTY_PIXELS_PER_WINDOW: str(self._pixels_per_window),
            configuration.PROPERTY_SAMPLING_SIZE: str(self._sampling_size),
            configuration.PROPERTY_R2_THRESHOLD: str(self._r2_threshold)
        }
        with open(self._config_file, 'w') as configfile:
            test_configuration.write(configfile)
        self.assertRaises(
            configuration.ConfigurationException, configuration.Configuration, self._config_file)