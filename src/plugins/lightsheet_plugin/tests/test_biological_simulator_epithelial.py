#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:54:21 2024

@author: george
"""

import unittest
import numpy as np
from biological_simulation import BiologicalSimulator

class TestBiologicalSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = BiologicalSimulator(size=(30, 100, 100), num_time_points=10)

    def test_epithelial_cell_generation(self):
        cell_shape, cell_interior, cell_membrane = self.simulator.generate_cell_shape(
            cell_type='epithelial',
            size=(30, 100, 100),
            pixel_size=(1, 1, 1),
            membrane_thickness=1,
            height=10
        )

        # Check shapes
        self.assertEqual(cell_shape.shape, (30, 100, 100))
        self.assertEqual(cell_interior.shape, (30, 100, 100))
        self.assertEqual(cell_membrane.shape, (30, 100, 100))

        # Check data types
        self.assertEqual(cell_shape.dtype, np.float64)
        self.assertEqual(cell_interior.dtype, np.float64)
        self.assertEqual(cell_membrane.dtype, np.float64)

        # Check cell height
        self.assertTrue(np.any(cell_shape[9]))
        self.assertFalse(np.any(cell_shape[10]))

        # Check membranes
        self.assertTrue(np.all(cell_membrane[0]))  # Basal membrane
        self.assertTrue(np.all(cell_membrane[9]))  # Apical membrane
        self.assertTrue(np.all(cell_membrane[:10, 0, :]))  # Lateral membrane

        # Check interior
        self.assertTrue(np.any(cell_interior[1:9, 1:-1, 1:-1]))
        self.assertFalse(np.any(cell_interior[0]))
        self.assertFalse(np.any(cell_interior[9]))

if __name__ == '__main__':
    unittest.main()
