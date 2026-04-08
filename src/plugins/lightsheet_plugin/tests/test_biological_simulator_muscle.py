import unittest
import numpy as np
from biological_simulation import BiologicalSimulator

class TestBiologicalSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = BiologicalSimulator(size=(30, 100, 100), num_time_points=10)

    def test_generate_cell_shape_muscle(self):
        cell_type = 'muscle'
        size = (30, 100, 100)
        pixel_size = (1, 1, 1)
        membrane_thickness = 1

        cell_shape, cell_interior, cell_membrane = self.simulator.generate_cell_shape(
            cell_type, size, pixel_size, membrane_thickness
        )

        # Assert that the shapes are correct
        self.assertEqual(cell_shape.shape, size)
        self.assertEqual(cell_interior.shape, size)
        self.assertEqual(cell_membrane.shape, size)

        # Assert that the cell shape is not empty
        self.assertTrue(np.any(cell_shape))

        # Assert that the membrane is the difference between shape and interior
        np.testing.assert_array_equal(cell_membrane, (cell_shape > 0) & (cell_interior == 0))

    def test_generate_cell_shape_muscle_with_array_input(self):
        cell_type = 'muscle'
        size = (np.array([30]), np.array([100]), np.array([100]))
        pixel_size = (1, 1, 1)
        membrane_thickness = 1

        cell_shape, cell_interior, cell_membrane = self.simulator.generate_cell_shape(
            cell_type, size, pixel_size, membrane_thickness
        )

        # Assert that the shapes are correct
        self.assertEqual(cell_shape.shape, (30, 100, 100))
        self.assertEqual(cell_interior.shape, (30, 100, 100))
        self.assertEqual(cell_membrane.shape, (30, 100, 100))

        # Assert that the cell shape is not empty
        self.assertTrue(np.any(cell_shape))

        # Assert that the membrane is the difference between shape and interior
        np.testing.assert_array_equal(cell_membrane, (cell_shape > 0) & (cell_interior == 0))

if __name__ == '__main__':
    unittest.main()
