import unittest
import numpy as np
from scipy.ndimage import label, binary_dilation
from biological_simulation import BiologicalSimulator  # Make sure to import from the correct module

class TestBiologicalSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = BiologicalSimulator(size=(100, 100, 100), num_time_points=1)

    def test_generate_er(self):
        # Create a simple cell shape and nucleus
        cell_shape = np.ones(self.simulator.size, dtype=bool)
        soma_center = tuple(s // 2 for s in self.simulator.size)
        nucleus_radius = 20

        er = self.simulator.generate_er(cell_shape, soma_center, nucleus_radius)

        # Check if the output is a float array of the correct shape
        self.assertEqual(er.shape, self.simulator.size)
        self.assertEqual(er.dtype, float)

        # Check if ER is present and within reasonable volume range (3-10% of cytoplasm volume)
        nucleus_mask = np.sum((np.indices(er.shape) - np.array(soma_center)[:, None, None, None])**2, axis=0) <= nucleus_radius**2
        cytoplasm_mask = cell_shape & ~nucleus_mask
        er_volume = np.sum(er > 0)
        cytoplasm_volume = np.sum(cytoplasm_mask)
        er_fraction = er_volume / cytoplasm_volume
        self.assertGreater(er_fraction, 0.03)
        self.assertLess(er_fraction, 0.10)

        # Check if ER forms a connected network
        labeled_er, num_features = label(er > 0)
        largest_component = np.sum(labeled_er == np.argmax(np.bincount(labeled_er.flat)[1:]) + 1)
        connectivity_ratio = largest_component / er_volume
        self.assertGreater(connectivity_ratio, 0.5)  # At least 50% should be connected

        # Check if ER density is higher near the nucleus
        near_nucleus = (nucleus_mask == 0) & (binary_dilation(nucleus_mask, iterations=5) > 0)
        far_from_nucleus = ~binary_dilation(nucleus_mask, iterations=10)
        near_density = np.sum(er[near_nucleus]) / np.sum(near_nucleus)
        far_density = np.sum(er[far_from_nucleus]) / np.sum(far_from_nucleus)
        self.assertGreater(near_density, far_density * 0.9)  # Near density should be at least 90% of far density

        # Check if ER doesn't enter the nucleus
        self.assertEqual(np.sum(er[nucleus_mask]), 0)

if __name__ == '__main__':
    unittest.main()
