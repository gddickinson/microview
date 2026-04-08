#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:49:14 2024

@author: george
"""

import unittest
import numpy as np
from lightsheetViewer import LightsheetViewer

class TestLightsheetViewer(unittest.TestCase):
    def setUp(self):
        self.viewer = LightsheetViewer()

    def test_addDataChannel(self):
        # Test initialization
        initial_data = np.zeros((10, 1, 30, 100, 100))
        self.viewer.addDataChannel(initial_data, "Initial")
        self.assertEqual(self.viewer.data.shape, (10, 1, 30, 100, 100))

        # Test adding a channel
        new_channel = np.ones((10, 1, 30, 100, 100))
        self.viewer.addDataChannel(new_channel, "New Channel")
        self.assertEqual(self.viewer.data.shape, (10, 2, 30, 100, 100))

        # Test adding a channel with incorrect shape
        incorrect_channel = np.ones((10, 1, 30, 100, 50))
        with self.assertRaises(ValueError):
            self.viewer.addDataChannel(incorrect_channel, "Incorrect Channel")

if __name__ == '__main__':
    unittest.main()
