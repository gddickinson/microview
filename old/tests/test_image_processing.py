#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 14:14:25 2024

@author: george
"""

# tests/test_image_processing.py

import unittest
import numpy as np
from microview.image_processing import ImageProcessor

class TestImageProcessor(unittest.TestCase):

    def setUp(self):
        self.test_image = np.random.rand(10, 10)
        self.test_stack = np.random.rand(5, 10, 10)

    def test_gaussian_blur(self):
        blurred = ImageProcessor.gaussian_blur(self.test_image, 1.0)
        self.assertEqual(blurred.shape, self.test_image.shape)
        self.assertRaises(ValueError, ImageProcessor.gaussian_blur, self.test_image, -1)

    def test_median_filter(self):
        filtered = ImageProcessor.median_filter(self.test_image, 3)
        self.assertEqual(filtered.shape, self.test_image.shape)
        self.assertRaises(ValueError, ImageProcessor.median_filter, self.test_image, 2)

    def test_threshold(self):
        thresholded = ImageProcessor.threshold(self.test_image)
        self.assertEqual(thresholded.dtype, bool)

    def test_z_project(self):
        max_projected = ImageProcessor.z_project(self.test_stack, 'max')
        mean_projected = ImageProcessor.z_project(self.test_stack, 'mean')
        self.assertEqual(max_projected.shape, self.test_stack.shape[1:])
        self.assertEqual(mean_projected.shape, self.test_stack.shape[1:])
        self.assertRaises(ValueError, ImageProcessor.z_project, self.test_stack, 'invalid')

if __name__ == '__main__':
    unittest.main()
