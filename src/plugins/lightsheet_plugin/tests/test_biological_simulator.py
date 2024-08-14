#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:39:59 2024

@author: george
"""

import unittest
import numpy as np
from biological_simulation import Cell, CellularEnvironment, EnhancedBiologicalSimulator

class TestCell(unittest.TestCase):
    def setUp(self):
        self.cell_size = (30, 100, 100)
        self.cell_position = (0, 0, 0)
        self.cell = Cell(self.cell_size, self.cell_position)
        self.cell.initialize_structures()

    def test_cell_initialization(self):
        self.assertEqual(self.cell.size, self.cell_size)
        self.assertEqual(self.cell.position, self.cell_position)
        self.assertIsNotNone(self.cell.membrane)
        self.assertIsNotNone(self.cell.nucleus)
        self.assertIsNotNone(self.cell.cytoplasm)

    def test_add_protein(self):
        protein_name = "test_protein"
        initial_concentration = np.zeros(self.cell_size)
        initial_concentration[15, 50, 50] = 1  # Single point source
        diffusion_coefficient = 0.1

        self.cell.add_protein(protein_name, initial_concentration, diffusion_coefficient)

        self.assertIn(protein_name, self.cell.proteins)
        self.assertIn(protein_name, self.cell.diffusion_coefficients)
        np.testing.assert_array_equal(self.cell.proteins[protein_name], initial_concentration)
        self.assertEqual(self.cell.diffusion_coefficients[protein_name], diffusion_coefficient)

    def test_protein_diffusion(self):
        protein_name = "test_protein"
        initial_concentration = np.zeros(self.cell.size)
        initial_concentration[15, 50, 50] = 1  # Single point source
        diffusion_coefficient = 0.1

        self.cell.add_protein(protein_name, initial_concentration, diffusion_coefficient)

        print("Initial protein distribution:")
        print(self.cell.proteins[protein_name].sum(axis=2))
        print("Total initial protein:", np.sum(self.cell.proteins[protein_name]))

        # Run diffusion for one time step
        dt = 1.0
        self.cell.update_protein_diffusion(dt)

        # Check that the protein has diffused
        diffused_concentration = self.cell.proteins[protein_name]
        non_zero_count = np.sum(diffused_concentration > 0)

        print("Diffused protein distribution:")
        print(diffused_concentration.sum(axis=2))
        print("Total protein after diffusion:", np.sum(diffused_concentration))

        self.assertGreater(non_zero_count, 1, f"Expected more than 1 non-zero value, but got {non_zero_count}")
        print(f"Non-zero concentration points: {non_zero_count}")

class TestCellularEnvironment(unittest.TestCase):
    def setUp(self):
        self.env_size = (50, 150, 150)
        self.environment = CellularEnvironment(self.env_size)

        cell_size = (30, 100, 100)
        cell1 = Cell(cell_size, (0, 0, 0))
        cell2 = Cell(cell_size, (20, 50, 50))
        cell1.initialize_structures()
        cell2.initialize_structures()

        self.environment.add_cell(cell1)
        self.environment.add_cell(cell2)

    def test_environment_initialization(self):
        self.assertEqual(self.environment.size, self.env_size)
        self.assertEqual(len(self.environment.cells), 2)
        self.assertTrue(np.any(self.environment.extracellular_space))

    def test_add_global_protein(self):
        protein_name = "test_global_protein"
        initial_concentration = np.zeros(self.env_size)
        initial_concentration[25, 75, 75] = 1  # Single point source
        diffusion_coefficient = 0.1

        self.environment.add_global_protein(protein_name, initial_concentration, diffusion_coefficient)

        self.assertIn(protein_name, self.environment.global_proteins)
        np.testing.assert_array_equal(self.environment.global_proteins[protein_name]['concentration'], initial_concentration)
        self.assertEqual(self.environment.global_proteins[protein_name]['diffusion_coefficient'], diffusion_coefficient)


    def test_cell_cell_interactions(self):
        initial_positions = [cell.position for cell in self.environment.cells]
        print("Initial cell positions:", initial_positions)

        moved = False
        for _ in range(10):  # Run multiple updates
            moved |= self.environment.update(1.0)
            if moved:
                break

        final_positions = [cell.position for cell in self.environment.cells]
        print("Final cell positions:", final_positions)

        self.assertTrue(moved, "No cell positions changed after multiple interactions")
        if moved:
            print("Cell positions changed as expected")
        else:
            print("Cell positions did not change")

    def test_update(self):
        # Add a global protein
        protein_name = "test_global_protein"
        initial_concentration = np.zeros(self.env_size)
        initial_concentration[25, 75, 75] = 1  # Single point source
        self.environment.add_global_protein(protein_name, initial_concentration, 0.1)

        # Add proteins to cells
        for cell in self.environment.cells:
            cell_protein = np.zeros(cell.size)
            cell_protein[15, 50, 50] = 1  # Single point source
            cell.add_protein("cell_protein", cell_protein, 0.1)

        # Run update
        self.environment.update(1.0)

        # Check that global protein has diffused
        global_protein_sum = np.sum(self.environment.global_proteins[protein_name]['concentration'] > 0)
        self.assertGreater(global_protein_sum, 1, f"Expected more than 1 non-zero value for global protein, but got {global_protein_sum}")

        # Check that cell proteins have diffused
        for i, cell in enumerate(self.environment.cells):
            cell_protein_sum = np.sum(cell.proteins["cell_protein"] > 0)
            self.assertGreater(cell_protein_sum, 1, f"Expected more than 1 non-zero value for cell {i} protein, but got {cell_protein_sum}")

class TestEnhancedBiologicalSimulator(unittest.TestCase):
    def setUp(self):
        self.sim_size = (50, 150, 150)
        self.num_time_points = 10
        self.simulator = EnhancedBiologicalSimulator(self.sim_size, self.num_time_points)

    def test_add_cell(self):
        cell_size = (30, 100, 100)
        initial_cell_count = len(self.simulator.environment.cells)
        self.simulator.add_cell((0, 0, 0), cell_size)
        self.assertEqual(len(self.simulator.environment.cells), initial_cell_count + 1)
        self.assertEqual(self.simulator.environment.cells[-1].size, cell_size)

    def test_simulator_initialization(self):
        self.assertEqual(self.simulator.environment.size, self.sim_size)
        self.assertEqual(self.simulator.num_time_points, self.num_time_points)
        self.assertEqual(self.simulator.current_time, 0)


    def test_add_global_protein(self):
        protein_name = "test_global_protein"
        initial_concentration = np.zeros(self.sim_size)
        initial_concentration[25, 75, 75] = 1  # Single point source
        diffusion_coefficient = 0.1

        self.simulator.add_global_protein(protein_name, initial_concentration, diffusion_coefficient)

        self.assertIn(protein_name, self.simulator.environment.global_proteins)
        np.testing.assert_array_equal(
            self.simulator.environment.global_proteins[protein_name]['concentration'],
            initial_concentration
        )

    def test_add_protein_to_cell(self):
        cell_size = (30, 100, 100)
        cell_position = (0, 0, 0)
        self.simulator.add_cell(cell_position, cell_size)

        protein_name = "test_cell_protein"
        initial_concentration = np.zeros(cell_size)
        initial_concentration[15, 50, 50] = 1  # Single point source
        diffusion_coefficient = 0.1

        self.simulator.add_protein_to_cell(0, protein_name, initial_concentration, diffusion_coefficient)

        self.assertIn(protein_name, self.simulator.environment.cells[0].proteins)
        np.testing.assert_array_equal(
            self.simulator.environment.cells[0].proteins[protein_name],
            initial_concentration
        )

    def test_run_simulation(self):
        # Add a cell and a protein
        cell_size = (30, 100, 100)
        cell_position = (0, 0, 0)
        self.simulator.add_cell(cell_position, cell_size)

        protein_name = "test_protein"
        initial_concentration = np.zeros(cell_size)
        initial_concentration[15, 50, 50] = 1  # Single point source
        diffusion_coefficient = 0.1

        self.simulator.add_protein_to_cell(0, protein_name, initial_concentration, diffusion_coefficient)

        # Run simulation
        states = list(self.simulator.run_simulation())

        # Check if we got the correct number of states
        self.assertEqual(len(states), self.num_time_points)

        # Check if the protein has diffused
        first_state = states[0]
        last_state = states[-1]
        first_sum = np.sum(first_state['proteins'][protein_name] > 0)
        last_sum = np.sum(last_state['proteins'][protein_name] > 0)
        self.assertGreater(last_sum, first_sum,
                           f"Expected more non-zero values in the last state ({last_sum}) compared to the first state ({first_sum})")
        print(f"Non-zero concentration points: first state = {first_sum}, last state = {last_sum}")

if __name__ == '__main__':
    unittest.main()
