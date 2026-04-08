#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:47:59 2024

@author: george
"""

# microview_utils.py

import numpy as np
from global_vars import g

class MicroViewImageHandler:
    @staticmethod
    def open_file(file_path):
        # Assuming MicroView has a method to open files and return a Window object
        window = g.m.open_file(file_path)
        return window.image

    @staticmethod
    def get_roi_trace(window, roi):
        # Assuming MicroView has a method to get ROI traces
        return window.get_roi_trace(roi)

class MicroViewROIHandler:
    @staticmethod
    def open_rois(file_path):
        # Assuming MicroView has a method to open ROI files
        return g.m.open_rois(file_path)

    @staticmethod
    def create_roi(x, y, width, height):
        # Assuming MicroView has a method to create ROIs
        return g.m.create_roi(x, y, width, height)
