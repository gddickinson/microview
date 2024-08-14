import unittest
import numpy as np
from biological_simulation import BiologicalSimulator
from scipy.ndimage import label, center_of_mass
import logging

logging.basicConfig(level=logging.DEBUG)

class TestBiologicalSimulator(unittest.TestCase):
    def setUp(self):
        self.size = (30, 100, 100)
        self.num_time_points = 10
        self.simulator = BiologicalSimulator(self.size, self.num_time_points)

        # Create a simple cell shape and nucleus for testing
        self.simulator.cell_shape = np.ones(self.size, dtype=bool)
        self.simulator.nucleus = np.zeros(self.size, dtype=bool)
        self.simulator.nucleus[10:20, 40:60, 40:60] = True

        self.logger = logging.getLogger(__name__)

    def test_generate_mitochondria(self):
        num_mitochondria = 50
        size_range = (3, 8)
        mitochondria = self.simulator.generate_mitochondria(num_mitochondria, size_range)

        self.assertEqual(mitochondria.shape, self.size)
        self.assertTrue(np.any(mitochondria))  # Check if any mitochondria were generated
        self.assertTrue(np.all(mitochondria[self.simulator.nucleus] == 0))  # Check if mitochondria avoid nucleus
        self.assertLessEqual(np.sum(mitochondria), np.sum(self.simulator.cell_shape))  # Check if mitochondria are within cell

        self.logger.info(f"Generated mitochondria volume: {np.sum(mitochondria)}")

    def test_generate_cytoskeleton(self):
        actin_density = 0.05
        microtubule_density = 0.02
        actin, microtubules = self.simulator.generate_cytoskeleton(actin_density, microtubule_density)

        self.assertEqual(actin.shape, self.size)
        self.assertEqual(microtubules.shape, self.size)
        self.assertTrue(np.any(actin))  # Check if any actin was generated
        self.assertTrue(np.any(microtubules))  # Check if any microtubules were generated
        self.assertTrue(np.all(actin[self.simulator.nucleus] == 0))  # Check if actin avoids nucleus
        self.assertTrue(np.all(microtubules[self.simulator.nucleus] == 0))  # Check if microtubules avoid nucleus

        self.logger.info(f"Generated actin volume: {np.sum(actin)}")
        self.logger.info(f"Generated microtubules volume: {np.sum(microtubules)}")

    def test_simulate_active_transport(self):
        # First, generate cytoskeleton
        self.simulator.generate_cytoskeleton()

        velocity = (1, 1, 1)
        initial_cargo = np.zeros(self.size)
        initial_cargo[15, 50, 50] = 1.0  # Place initial cargo in the center

        # Test with microtubules
        transport_data_mt = self.simulator.simulate_active_transport(velocity, initial_cargo, use_microtubules=True)
        self.assertEqual(transport_data_mt.shape, (self.num_time_points, *self.size))
        self.assertTrue(np.any(transport_data_mt))  # Check if any transport occurred

        self.logger.info(f"Microtubule transport: initial cargo volume = {np.sum(transport_data_mt[0])}, final cargo volume = {np.sum(transport_data_mt[-1])}")

        # Test with actin
        transport_data_actin = self.simulator.simulate_active_transport(velocity, initial_cargo, use_microtubules=False)
        self.assertEqual(transport_data_actin.shape, (self.num_time_points, *self.size))
        self.assertTrue(np.any(transport_data_actin))  # Check if any transport occurred

        self.logger.info(f"Actin transport: initial cargo volume = {np.sum(transport_data_actin[0])}, final cargo volume = {np.sum(transport_data_actin[-1])}")

    def test_mitochondria_size_range(self):
        num_mitochondria = 100
        size_range = (3, 8)
        mitochondria = self.simulator.generate_mitochondria(num_mitochondria, size_range)

        # Find connected components in the mitochondria array
        labeled, num_features = label(mitochondria)

        # Calculate the size of each component
        component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]

        # Check if all component sizes are within the specified range
        min_size = 4/3 * np.pi * (size_range[0]/2)**3 * 0.4  # Allow 60% margin for discretization
        max_size = 4/3 * np.pi * (size_range[1]/2)**3 * 1.6  # Allow 60% margin for discretization

        self.logger.info(f"Mitochondria size range: min_size = {min_size}, max_size = {max_size}")
        self.logger.info(f"Actual mitochondria sizes: {component_sizes}")

        self.assertTrue(all(min_size <= size <= max_size for size in component_sizes))

    def test_cytoskeleton_density(self):
        actin_density = 0.05
        microtubule_density = 0.02
        actin, microtubules = self.simulator.generate_cytoskeleton(actin_density, microtubule_density)

        # Calculate the actual densities
        cytoplasm_volume = np.sum(self.simulator.cell_shape) - np.sum(self.simulator.nucleus)
        actual_actin_density = np.sum(actin) / cytoplasm_volume
        actual_microtubule_density = np.sum(microtubules) / cytoplasm_volume

        self.logger.info(f"Actin density: target = {actin_density}, actual = {actual_actin_density}")
        self.logger.info(f"Microtubule density: target = {microtubule_density}, actual = {actual_microtubule_density}")

        # Check if the actual densities are close to the specified densities
        self.assertLess(abs(actual_actin_density - actin_density), 0.03)
        self.assertLess(abs(actual_microtubule_density - microtubule_density), 0.03)

    def test_active_transport_direction(self):
        self.simulator.generate_cytoskeleton()

        velocity = (1, 0, 0)  # Transport along x-axis
        initial_cargo = np.zeros(self.size)
        initial_cargo[15, 50, 0] = 1.0  # Place initial cargo at the left edge

        transport_data = self.simulator.simulate_active_transport(velocity, initial_cargo)

        # Check if the cargo has moved to the right over time
        self.assertTrue(np.any(transport_data[-1]))  # Ensure some cargo remains at the end

        initial_x = np.mean(np.where(transport_data[0] > 0)[2]) if np.any(transport_data[0]) else 0
        final_x = np.mean(np.where(transport_data[-1] > 0)[2]) if np.any(transport_data[-1]) else 0

        self.logger.info(f"Active transport: initial_x = {initial_x}, final_x = {final_x}")

        self.assertGreaterEqual(final_x, initial_x)  # x-coordinate should not decrease

if __name__ == '__main__':
    unittest.main()
