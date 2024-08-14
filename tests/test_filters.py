#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:21:52 2024

@author: george
"""

# tests/test_filters.py

import unittest
import numpy as np
from unittest.mock import MagicMock
from microview.filters import Filters

class TestFilters(unittest.TestCase):

    def setUp(self):
        self.mock_parent = MagicMock()
        self.mock_parent.current_window.image = np.random.rand(10, 10)
        self.filters = Filters(self.mock_parent)

    def test_apply_filter(self):
        def dummy_filter(image):
            return image * 2

        self.filters.apply_filter(dummy_filter)
        self.mock_parent.loadImage.assert_called_once()

    def test_apply_filter_no_image(self):
        self.mock_parent.current_window = None
        with self.assertRaises(ValueError):
            self.filters.apply_filter(lambda x: x)

    def test_sobel(self):
        self.filters.sobel()
        self.mock_parent.loadImage.assert_called_once()

    # Add more tests for other filter methods...

if __name__ == '__main__':
    unittest.main()
