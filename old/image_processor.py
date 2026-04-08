#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 08:32:42 2024

@author: george
"""

# image_processor.py
from scipy import ndimage
from skimage import filters, morphology
import numpy as np

class ImageProcessor:
    @staticmethod
    def gaussian_blur(image, sigma):
        return ndimage.gaussian_filter(image, sigma)

    @staticmethod
    def median_filter(image, size):
        return ndimage.median_filter(image, size)

    @staticmethod
    def sobel_edge(image):
        return filters.sobel(image)

    @staticmethod
    def threshold(image):
        threshold = filters.threshold_otsu(image)
        return image > threshold

    @staticmethod
    def erode(image):
        return morphology.erosion(image)

    @staticmethod
    def dilate(image):
        return morphology.dilation(image)

    @staticmethod
    def z_project_max(image):
        return np.max(image, axis=0)

    @staticmethod
    def z_project_mean(image):
        return np.mean(image, axis=0)
