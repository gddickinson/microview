#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:12:12 2024

@author: george
"""

import numpy as np
from scipy import ndimage

class ImageProcessor:
    def gaussian_blur(self, data, sigma=1):
        if data.ndim == 5:  # t, c, z, y, x
            return np.array([[[ndimage.gaussian_filter(slice, sigma) for slice in channel] for channel in volume] for volume in data])
        elif data.ndim == 4:  # t, z, y, x
            return np.array([[ndimage.gaussian_filter(slice, sigma) for slice in volume] for volume in data])
        elif data.ndim == 3:  # z, y, x
            return np.array([ndimage.gaussian_filter(slice, sigma) for slice in data])

    def median_filter(self, data, size=3):
        if data.ndim == 5:  # t, c, z, y, x
            return np.array([[[ndimage.median_filter(slice, size) for slice in channel] for channel in volume] for volume in data])
        elif data.ndim == 4:  # t, z, y, x
            return np.array([[ndimage.median_filter(slice, size) for slice in volume] for volume in data])
        elif data.ndim == 3:  # z, y, x
            return np.array([ndimage.median_filter(slice, size) for slice in data])

    # Add more image processing methods here...
