#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:29:26 2024

@author: george
"""

# test_biological_data_generation.py

import unittest
import numpy as np
from biological_data_generation import BiologicalDataGenerator

class TestBiologicalDataGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = BiologicalDataGenerator()

    def test_generate_volume(self):
        params = {
            'size': (50, 50, 50),
            'cell_radius': 20
        }
        volume = self.generator.generate_volume(params)
        self.assertEqual(volume.shape, params['size'])
        self.assertTrue(0 <= volume.min() <= volume.max() <= 1)

        # Check for presence of cellular structures
        self.assertTrue(np.any(volume == 0.8))  # membrane
        self.assertTrue(np.any(volume == 0.6))  # nucleus
        self.assertTrue(np.any(volume == 0.4))  # cytoplasm

    def test_generate_time_series(self):
        params = {
            'num_volumes': 5,
            'size': (50, 50, 50),
            'cell_radius': 20
        }
        time_series = self.generator.generate_time_series(params)
        self.assertEqual(time_series.shape, (params['num_volumes'], *params['size']))
        self.assertTrue(0 <= time_series.min() <= time_series.max() <= 1)

if __name__ == '__main__':
    unittest.main()
