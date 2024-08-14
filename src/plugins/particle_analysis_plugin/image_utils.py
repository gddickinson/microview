#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:32:32 2024

@author: george
"""

# image_utils.py

import numpy as np
from skimage import io

class ImageHandler:
    @staticmethod
    def open_file(file_path):
        return io.imread(file_path)

    @staticmethod
    def get_roi_trace(image_data, roi):
        # This is a simplified version. You'll need to implement
        # the actual ROI trace extraction based on your needs.
        return np.mean(image_data[:, roi[0]:roi[2], roi[1]:roi[3]], axis=(1, 2))

class ROIHandler:
    @staticmethod
    def open_rois(file_path):
        # Implement a method to read ROI files in the format you're using
        # This is just a placeholder
        return [{'x': 0, 'y': 0, 'width': 10, 'height': 10}]
