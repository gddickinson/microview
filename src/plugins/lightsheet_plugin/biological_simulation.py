#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 07:09:57 2024

@author: george
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Any, Dict
from skimage.morphology import skeletonize
from scipy.ndimage import center_of_mass, binary_dilation, gaussian_filter
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label
from skimage.morphology import skeletonize
from scipy.spatial import  cKDTree
from scipy.spatial.transform import Rotation
from abc import ABC, abstractmethod


'''Basic organelle info'''
'''Within the cytoplasm, the major organelles and cellular structures include: (1) nucleolus (2) nucleus (3) ribosome (4) vesicle (5) rough endoplasmic reticulum (6) Golgi apparatus (7) cytoskeleton (8) smooth endoplasmic reticulum (9) mitochondria (10) vacuole (11) cytosol (12) lysosome (13) centriole'''

'''The endoplasmic reticulum (ER) is a netlike labyrinth of tubules and sacs that extends throughout the cytosol of a cell. It's interconnected with other organelles, such as the Golgi apparatus, mitochondria, and nucleus, through membrane contacts and vesicle transport. These connections allow for the exchange of materials and signals between different cellular compartments.'''

'''The nucleus of a cell is surrounded by a nuclear envelope, which is made up of two concentric membranes: the inner and outer nuclear membranes. The space between the two membranes is called the perinuclear space, and it's directly connected to the endoplasmic reticulum's lumen:
Outer nuclear membrane
Continuous with the endoplasmic reticulum, this membrane has ribosomes attached to its cytoplasmic surface and is functionally similar to the endoplasmic reticulum's membranes.
Inner nuclear membrane
Surrounds the nucleus's contents and has unique proteins attached to it that are specific to the nucleus. Embedded within the inner membrane are proteins that bind to intermediate filaments, which give the nucleus its structure.'''

'''Mitochondria are organelles that interact with other organelles in the cell to regulate energy metabolism, biosynthesis, immune response, and cell turnover. These interactions occur through membrane contact sites (MCSs), signal transduction, and vesicle transport. Mitochondria form MCSs with almost every other organelle, including:
Endoplasmic reticulum (ER), Lipid droplets, Lysosomes, Golgi apparatus, Melanosomes, Peroxisomes, Cytoskeleton, and Nucleus.'''

'''The plasma membrane is a semi-permeable barrier that separates the inside and outside of a cell, and it's made up of lipids and proteins. The membrane's fundamental structure is a phospholipid bilayer, which is made up of lipid molecules with two fatty acid chains and a phosphate-containing group. Proteins are embedded within the bilayer and perform specific functions, such as transporting molecules and recognizing other cells. The plasma membrane also connects to other organelles through vesicular transport, which exchanges membrane components like proteins and lipids. Molecular tags help direct the components to their proper destinations. For example, the endoplasmic reticulum (ER) is a large, continuous membrane-bound organelle that's found in the cytoplasm of eukaryotic cells. The ER has many distinct domains, including the plasma membrane, and it can form abundant contacts with the plasma membrane in cell bodies, as well as smaller contacts in other areas. These contacts are bridged by tethering molecules that can be constant or dynamically regulated to perform different functions.'''

'''The Golgi apparatus interacts with other organelles in the cell through molecules that are transported into or out of it. The Golgi is part of the endomembrane system, which also includes the endoplasmic reticulum (ER) and lysosomes, and these organelles have close relationships with the Golgi:
Lysosomes
The Golgi is responsible for forming lysosomes when vesicles bud off from the trans-Golgi and fuse with endosomes. The ER also contributes to lysosomal formation by synthesizing lysosomal hydrolases, which are then transported to the Golgi and tagged for the lysosomes. Golgi–lysosome contact may regulate signaling and the comovement of these organelles, and their interplay may play a role in cancer and neurodegenerative diseases.
Endosomes
The Golgi sorts newly synthesized proteins for transport to other endomembrane organelles, including endosomes.
ER
The Golgi and ER have a bidirectional relationship that helps coordinate signal transduction between organelles during cell remodeling.
N-ethylmaleimide-sensitive factor (NSF)
NSF is closely related to endothelial nitric oxide synthase (eNOS), which is mainly located in the Golgi apparatus. NSF reduces the speed of protein transport from the Golgi to the plasma membrane.'''

'''The cytoskeleton is made up of microtubules, actin filaments, and intermediate filaments, which form an interconnected network. This network can exert and transmit mechanical forces to other cellular components, such as organelles, which can modify their function and morphology. The cytoskeleton also remodels membrane-bound organelles, and in turn, dynamic membrane-bound organelles can contribute to cytoskeletal organization. These complex interactions between the cytoskeleton and organelles play important roles in their transport, organization, and dynamics, which are essential for maintaining cellular homeostasis. Microtubules
These can connect to other microtubules or organelles using lateral appendages. Microtubule motors can also attach to vesicles and move along microtubules to pull them.
Microfilaments
These help structure the cytoplasm and movement. Proteins can bind to actin filaments to regulate their formation and function.
Intermediate filaments
These have structural functions, such as bearing tension to maintain cell shape and anchoring organelles in place. '''

'''Ribosomes are organelles that can interact with other organelles in several ways, including:
Endoplasmic reticulum (ER)
When ribosomes synthesize proteins for the ER or for export from the cell, they can attach to the ER, giving it a rough appearance. This process is called vectorial synthesis, and the ribosomes bind to the ER's translocon site. The ribosomes are not permanent parts of the ER's structure, and they are constantly being released and reattached to the membrane.
Ribosome-associated vesicles (RAVs)
These organelles interact with mitochondria through direct membrane contact, which helps the ER and its derivatives communicate with other organelles.
Nuclear envelope
The nuclear envelope has a thin space between its two layers that's directly connected to the ER's interior. Small channels called nuclear pores span the nuclear envelope, allowing substances to pass in and out of the nucleus. '''


class Cell:
    def __init__(self, size: Tuple[int, int, int], position: Tuple[float, float, float]):
        self.size = size
        self.position = position
        self.membrane = np.zeros(size, dtype=bool)
        self.nucleus = np.zeros(size, dtype=bool)
        self.cytoplasm = np.zeros(size, dtype=bool)
        self.er = np.zeros(size, dtype=bool)
        self.proteins = {}
        self.diffusion_coefficients = {}
        self.initialize_structures()

    def initialize_structures(self):
        center = tuple(s // 2 for s in self.size)
        radius = min(self.size) // 3

        z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
        dist_from_center = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)

        # Create cell membrane
        self.membrane = (dist_from_center <= radius + 1) & (dist_from_center > radius)

        # Create cytoplasm
        self.cytoplasm = dist_from_center <= radius

        # Create nucleus (smaller than the cell)
        nucleus_radius = radius // 2
        self.nucleus = dist_from_center <= nucleus_radius

        # Create ER (between nucleus and cell membrane)
        er_mask = (dist_from_center > nucleus_radius + 1) & (dist_from_center < radius - 1)
        self.er = er_mask & (np.random.rand(*self.size) < 0.2)  # 20% density

    def update_protein_diffusion(self, dt: float):
        for name, concentration in self.proteins.items():
            D = self.diffusion_coefficients[name]
            sigma = np.sqrt(2 * D * dt)

            # Apply diffusion
            diffused = gaussian_filter(concentration, sigma=sigma, mode='constant', cval=0)

            # Ensure diffusion only occurs within the cell
            cell_mask = self.membrane | self.cytoplasm | self.nucleus
            self.proteins[name] = np.where(cell_mask, diffused, 0)

            # Normalize to maintain total protein amount
            total_protein = np.sum(self.proteins[name])
            if total_protein > 0:
                self.proteins[name] *= np.sum(concentration) / total_protein

    def update(self, dt: float):
        self.update_protein_diffusion(dt)
        # Other update methods will be added here later


class CellularEnvironment:
    def __init__(self, size: Tuple[int, int, int]):
        self.size = size
        self.cells: List[Cell] = []
        self.extracellular_space = np.ones(size, dtype=bool)
        self.global_proteins = {}

    def add_cell(self, cell: Cell):
        self.cells.append(cell)
        self._update_extracellular_space()

    def add_global_protein(self, name: str, initial_concentration: np.ndarray, diffusion_coefficient: float):
        if initial_concentration.shape != self.size:
            raise ValueError("Protein concentration array must match environment size")
        self.global_proteins[name] = {
            'concentration': initial_concentration,
            'diffusion_coefficient': diffusion_coefficient
        }

    def update(self, dt: float):
        self._update_extracellular_space()
        self._update_global_protein_diffusion(dt)
        for cell in self.cells:
            cell.update(dt)
        moved = self._cell_cell_adhesion() if self.cells else False
        print("Cell positions after update:", [cell.position for cell in self.cells] if self.cells else "No cells")
        return moved

    def _update_extracellular_space(self):
        self.extracellular_space.fill(1)
        for cell in self.cells:
            cell_space = np.zeros(self.size, dtype=bool)
            cell_space[
                cell.position[0]:cell.position[0]+cell.size[0],
                cell.position[1]:cell.position[1]+cell.size[1],
                cell.position[2]:cell.position[2]+cell.size[2]
            ] = cell.membrane | cell.cytoplasm
            self.extracellular_space &= ~cell_space

    def _update_global_protein_diffusion(self, dt: float):
        for name, protein_data in self.global_proteins.items():
            concentration = protein_data['concentration']
            D = protein_data['diffusion_coefficient']
            sigma = np.sqrt(2 * D * dt)

            diffused = gaussian_filter(concentration, sigma=sigma, mode='constant', cval=0)
            self.global_proteins[name]['concentration'] = np.where(self.extracellular_space, diffused, concentration)

    def _handle_cell_interactions(self):
        # Cell-cell adhesion
        self._cell_cell_adhesion()

        # Cell-cell signaling
        self._cell_cell_signaling()

    def _cell_cell_adhesion(self):
        cell_positions = np.array([cell.position for cell in self.cells])
        distances = cdist(cell_positions, cell_positions)

        adhesion_force = 1.0  # Start with a small force
        max_adhesion_distance = 60  # Increase the interaction distance

        moved = False
        for i, cell in enumerate(self.cells):
            nearby_cells = np.where((distances[i] > 0) & (distances[i] < max_adhesion_distance))[0]
            if len(nearby_cells) > 0:
                force_vectors = cell_positions[nearby_cells] - np.array(cell.position)
                total_force = np.sum(force_vectors * adhesion_force / (distances[i, nearby_cells][:, np.newaxis] + 1e-6), axis=0)
                new_position = np.array(cell.position) + total_force
                new_position = np.clip(new_position, 0, np.array(self.size) - np.array(cell.size) - 1)
                if not np.array_equal(new_position, cell.position):
                    cell.position = tuple(new_position.astype(int))
                    moved = True
            else:
                # If no nearby cells, move randomly
                random_move = np.random.randint(-1, 2, size=3)
                new_position = np.array(cell.position) + random_move
                new_position = np.clip(new_position, 0, np.array(self.size) - np.array(cell.size) - 1)
                if not np.array_equal(new_position, cell.position):
                    cell.position = tuple(new_position.astype(int))
                    moved = True

        print("Cell positions after adhesion:", [cell.position for cell in self.cells])
        return moved


    def _cell_cell_signaling(self):
        # Simple cell-cell signaling through extracellular space
        for cell in self.cells:
            for protein_name, protein_data in cell.proteins.items():
                if protein_name in self.global_proteins:
                    # Release some protein to extracellular space
                    release_rate = 0.01  # Arbitrary release rate
                    released_amount = release_rate * protein_data
                    cell.proteins[protein_name] -= released_amount

                    # Add released protein to global concentration
                    self.global_proteins[protein_name]['concentration'][
                        cell.position[0]:cell.position[0]+cell.size[0],
                        cell.position[1]:cell.position[1]+cell.size[1],
                        cell.position[2]:cell.position[2]+cell.size[2]
                    ] += released_amount

    def get_state(self):
        state = {
            'extracellular_space': self.extracellular_space,
            'global_proteins': {name: data['concentration'] for name, data in self.global_proteins.items()},
            'cells': [
                {
                    'position': cell.position,
                    'membrane': cell.membrane,
                    'nucleus': cell.nucleus,
                    'cytoplasm': cell.cytoplasm,
                    'proteins': cell.proteins
                } for cell in self.cells
            ]
        }
        return state


class EnhancedBiologicalSimulator:
    def __init__(self, size: Tuple[int, int, int], num_time_points: int):
        self.environment = CellularEnvironment(size)
        self.num_time_points = num_time_points
        self.current_time = 0
        self.dt = 1.0  # Time step, can be adjusted

    def add_cell(self, position: Tuple[float, float, float], cell_size: Tuple[int, int, int]):
        # Add some randomness to the initial position
        random_offset = np.random.randint(-5, 6, size=3)
        position = tuple(np.array(position) + random_offset)
        position = tuple(np.clip(position, 0, np.array(self.environment.size) - np.array(cell_size) - 1))

        cell = Cell(cell_size, position)
        cell.initialize_structures()
        self.environment.add_cell(cell)

    def add_global_protein(self, name: str, initial_concentration: np.ndarray, diffusion_coefficient: float):
        self.environment.add_global_protein(name, initial_concentration, diffusion_coefficient)

    def add_protein_to_cell(self, cell_index: int, name: str, initial_concentration: np.ndarray, diffusion_coefficient: float):
        if cell_index < 0 or cell_index >= len(self.environment.cells):
            raise ValueError("Invalid cell index")
        self.environment.cells[cell_index].add_protein(name, initial_concentration, diffusion_coefficient)

    def run_simulation(self):
        for t in range(self.num_time_points):
            print(f"Simulation step {t}")
            moved = self.environment.update(self.dt)
            state = self.get_current_state()
            for protein_name, concentration in state['proteins'].items():
                non_zero = np.sum(concentration > 0)
                total = np.sum(concentration)
                print(f"  Protein {protein_name}: {non_zero} non-zero points, total concentration: {total}")
            if self.environment.cells:
                print(f"  Cell positions: {[cell.position for cell in self.environment.cells]}")
            else:
                print("  No cells in the environment")
            print(f"  Cells moved: {moved}")
            yield state


    def get_current_state(self) -> Dict[str, np.ndarray]:
        state = self.environment.get_state()

        # Combine all cell data into single arrays for easier visualization
        combined_membrane = np.zeros(self.environment.size, dtype=bool)
        combined_nucleus = np.zeros(self.environment.size, dtype=bool)
        combined_cytoplasm = np.zeros(self.environment.size, dtype=bool)

        for cell_data in state['cells']:
            pos = cell_data['position']
            size = cell_data['membrane'].shape
            combined_membrane[
                pos[0]:pos[0]+size[0],
                pos[1]:pos[1]+size[1],
                pos[2]:pos[2]+size[2]
            ] |= cell_data['membrane']
            combined_nucleus[
                pos[0]:pos[0]+size[0],
                pos[1]:pos[1]+size[1],
                pos[2]:pos[2]+size[2]
            ] |= cell_data['nucleus']
            combined_cytoplasm[
                pos[0]:pos[0]+size[0],
                pos[1]:pos[1]+size[1],
                pos[2]:pos[2]+size[2]
            ] |= cell_data['cytoplasm']

        # Combine protein data
        protein_data = {}
        for cell_data in state['cells']:
            for protein_name, concentration in cell_data['proteins'].items():
                if protein_name not in protein_data:
                    protein_data[protein_name] = np.zeros(self.environment.size)
                pos = cell_data['position']
                size = concentration.shape
                protein_data[protein_name][
                    pos[0]:pos[0]+size[0],
                    pos[1]:pos[1]+size[1],
                    pos[2]:pos[2]+size[2]
                ] += concentration

        # Add global proteins
        for protein_name, concentration in state['global_proteins'].items():
            if protein_name not in protein_data:
                protein_data[protein_name] = concentration
            else:
                protein_data[protein_name] += concentration

        return {
            'membrane': combined_membrane,
            'nucleus': combined_nucleus,
            'cytoplasm': combined_cytoplasm,
            'extracellular_space': state['extracellular_space'],
            'proteins': protein_data
        }


def line_3d(x0, y0, z0, x1, y1, z1):
    """Generate coordinates of a 3D line using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1
    dm = max(dx, dy, dz)
    x, y, z = x0, y0, z0

    x_coords, y_coords, z_coords = [], [], []

    for _ in range(dm + 1):
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

        if x == x1 and y == y1 and z == z1:
            break

        if dx >= dy and dx >= dz:
            x += sx
            dy += dy
            dz += dz
            if dy >= dx:
                y += sy
                dy -= dx
            if dz >= dx:
                z += sz
                dz -= dx
        elif dy >= dx and dy >= dz:
            y += sy
            dx += dx
            dz += dz
            if dx >= dy:
                x += sx
                dx -= dy
            if dz >= dy:
                z += sz
                dz -= dy
        else:
            z += sz
            dx += dx
            dy += dy
            if dx >= dz:
                x += sx
                dx -= dz
            if dy >= dz:
                y += sy
                dy -= dz

    return np.array(x_coords), np.array(y_coords), np.array(z_coords)

class BiologicalSimulator:
    def __init__(self, size: Tuple[int, int, int], num_time_points: int):
        self.size = size
        self.num_time_points = num_time_points
        self.logger = logging.getLogger(__name__)
        self.cell_shape = None
        self.nucleus = None
        self.er = None
        self.mitochondria = None
        self.cytoskeleton = None

    def simulate_protein_diffusion(self, D: float, initial_concentration: np.ndarray) -> np.ndarray:
        try:
            if initial_concentration.shape != self.size:
                raise ValueError(f"Initial concentration shape {initial_concentration.shape} does not match simulator size {self.size}")

            result = np.zeros((self.num_time_points, *self.size))
            result[0] = initial_concentration

            for t in range(1, self.num_time_points):
                result[t] = gaussian_filter(result[t-1], sigma=np.sqrt(2*D))

            return result

        except Exception as e:
            self.logger.error(f"Error in protein diffusion simulation: {str(e)}")
            raise



    def generate_cellular_structure(self, structure_type: str) -> np.ndarray:
        """
        Generate a 3D representation of a cellular structure.

        Args:
        structure_type (str): Type of cellular structure ('nucleus', 'mitochondria', 'actin', 'lysosomes')

        Returns:
        np.ndarray: 3D array representing the cellular structure
        """
        try:
            if structure_type == 'nucleus':
                return self._generate_nucleus()
            elif structure_type == 'mitochondria':
                return self._generate_mitochondria()
            elif structure_type == 'actin':
                return self._generate_actin()
            elif structure_type == 'lysosomes':
                return self._generate_lysosomes()
            else:
                raise ValueError(f"Unknown structure type: {structure_type}")
        except Exception as e:
            self.logger.error(f"Error generating cellular structure: {str(e)}")
            raise

    def generate_cell_membrane(self, center, radius, thickness=1):
        """
        Generate a cell plasma membrane.

        Args:
        center (tuple): The (z, y, x) coordinates of the cell center
        radius (float): The radius of the cell
        thickness (float): The thickness of the membrane

        Returns:
        np.ndarray: 3D array representing the cell membrane
        """
        try:
            self.logger.info(f"Generating cell membrane at {center} with radius {radius}")

            z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
            dist_from_center = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)

            membrane = (dist_from_center >= radius - thickness/2) & (dist_from_center <= radius + thickness/2)

            self.logger.info("Cell membrane generated successfully")
            return membrane.astype(float)

        except Exception as e:
            self.logger.error(f"Error in cell membrane generation: {str(e)}")
            raise

    def generate_nucleus(self, cell_interior, soma_center, nucleus_radius, pixel_size=(1,1,1)):
        try:
            self.logger.info(f"Generating cell nucleus at {soma_center} with radius {nucleus_radius}")

            # Ensure nucleus_radius is at least 1 pixel and not larger than 1/3 of the soma
            soma_radius = min(cell_interior.shape) // 4
            nucleus_radius = min(max(1, nucleus_radius), soma_radius)

            # Find a suitable center for the nucleus
            new_center = self.find_suitable_center(cell_interior, soma_center, nucleus_radius)

            if new_center is None:
                self.logger.warning("Unable to find suitable location for nucleus.")
                return np.zeros_like(cell_interior), soma_center

            # Create a spherical nucleus
            z, y, x = np.ogrid[:cell_interior.shape[0], :cell_interior.shape[1], :cell_interior.shape[2]]
            dist_from_center = np.sqrt(
                ((z - new_center[0]) * pixel_size[0])**2 +
                ((y - new_center[1]) * pixel_size[1])**2 +
                ((x - new_center[2]) * pixel_size[2])**2
            )

            # Create nucleus only within the cell interior
            nucleus = (dist_from_center <= nucleus_radius) & (cell_interior > 0)

            actual_volume = np.sum(nucleus)
            self.nucleus = nucleus.astype(float)
            self.logger.info(f"Cell nucleus generated successfully. Nucleus volume: {actual_volume}")
            return nucleus.astype(float), new_center

        except Exception as e:
            self.logger.error(f"Error in cell nucleus generation: {str(e)}")
            raise

    def find_suitable_center(self, cell_shape, soma_center, nucleus_radius):
        z, y, x = np.ogrid[:cell_shape.shape[0], :cell_shape.shape[1], :cell_shape.shape[2]]
        dist_from_center = np.sqrt(
            (z - soma_center[0])**2 + (y - soma_center[1])**2 + (x - soma_center[2])**2
        )

        # Create a priority map that favors locations closer to the desired center
        priority_map = np.where(cell_shape > 0, -dist_from_center, -np.inf)

        search_radius = 0
        max_search_radius = max(cell_shape.shape)
        while search_radius < max_search_radius:
            potential_centers = np.argwhere(
                (dist_from_center <= search_radius) & (cell_shape > 0)
            )
            if len(potential_centers) > 0:
                # Sort potential centers by priority
                priorities = priority_map[tuple(potential_centers.T)]
                sorted_centers = potential_centers[np.argsort(priorities)]

                for center in sorted_centers:
                    # Check if the chosen center can accommodate the nucleus
                    z, y, x = np.ogrid[:cell_shape.shape[0], :cell_shape.shape[1], :cell_shape.shape[2]]
                    dist_from_center = np.sqrt(
                        (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
                    )
                    if np.sum((dist_from_center <= nucleus_radius) & (cell_shape > 0)) >= 7:
                        return center

            search_radius += 1

        # If no suitable center is found, return the center of mass of the cell shape
        if np.sum(cell_shape) > 0:
            return np.array(center_of_mass(cell_shape)).astype(int)
        else:
            return None


    def generate_er(self, cell_shape, soma_center, nucleus_radius, er_density=0.1, pixel_size=(1,1,1)):
        try:
            self.logger.info(f"Generating ER with density {er_density}")

            z, y, x = np.ogrid[:cell_shape.shape[0], :cell_shape.shape[1], :cell_shape.shape[2]]
            dist_from_center = np.sqrt(
                ((z - soma_center[0]) * pixel_size[0])**2 +
                ((y - soma_center[1]) * pixel_size[1])**2 +
                ((x - soma_center[2]) * pixel_size[2])**2
            )

            # Create a mask for the cell
            cell_mask = cell_shape > 0
            nucleus_mask = dist_from_center <= nucleus_radius
            cytoplasm_mask = cell_mask & ~nucleus_mask

            self.logger.info(f"Cell volume: {np.sum(cell_mask)}, Nucleus volume: {np.sum(nucleus_mask)}, Cytoplasm volume: {np.sum(cytoplasm_mask)}")

            # Initialize ER
            er = np.zeros_like(cell_shape, dtype=bool)

            # Generate ER throughout the cytoplasm
            er = cytoplasm_mask & (np.random.rand(*cell_shape.shape) < er_density)

            # Ensure higher ER density near the nucleus
            near_nucleus = (dist_from_center > nucleus_radius) & (dist_from_center <= nucleus_radius * 1.5)
            er |= near_nucleus & cytoplasm_mask & (np.random.rand(*cell_shape.shape) < er_density * 2)
            self.er = er
            self.logger.info(f"ER generated successfully. ER volume: {np.sum(er)}")
            return er.astype(float)

        except Exception as e:
            self.logger.error(f"Error in ER generation: {str(e)}")
            raise

    def generate_cell_shape(self, cell_type, size, pixel_size=(1,1,1), membrane_thickness=1, soma_center=None, **kwargs):
        try:
            self.logger.info(f"Generating {cell_type} cell shape")

            z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
            if soma_center is None:
                soma_center = np.array(size) // 2

            if cell_type == 'spherical':
                radius = kwargs.get('cell_radius', min(size) // 4)
                self.logger.info(f"Generating spherical cell with radius {radius}")
                dist_from_center = np.sqrt(
                    ((z - soma_center[0]) * pixel_size[0])**2 +
                    ((y - soma_center[1]) * pixel_size[1])**2 +
                    ((x - soma_center[2]) * pixel_size[2])**2
                )
                cell_interior = dist_from_center <= (radius - membrane_thickness)
                cell_membrane = (dist_from_center <= radius) & (dist_from_center > (radius - membrane_thickness))
                cell_shape = cell_interior | cell_membrane

            elif cell_type == 'neuron':
                soma_radius = kwargs.get('soma_radius', min(size) // 8)
                axon_length = kwargs.get('axon_length', size[2] // 2)
                axon_width = kwargs.get('axon_width', size[1] // 20)
                num_dendrites = kwargs.get('num_dendrites', 5)
                dendrite_length = kwargs.get('dendrite_length', size[1] // 4)

                self.logger.info(f"Generating neuron with soma radius {soma_radius}")

                # Create soma (cell body)
                dist_from_soma_center = np.sqrt(
                    ((z - soma_center[0]) * pixel_size[0])**2 +
                    ((y - soma_center[1]) * pixel_size[1])**2 +
                    ((x - soma_center[2]) * pixel_size[2])**2
                )
                soma = dist_from_soma_center <= soma_radius

                # Create axon
                axon_start = soma_center[2] + soma_radius
                axon = (x >= axon_start) & (x < axon_start + axon_length) & \
                       (np.abs(y - soma_center[1]) <= axon_width // 2) & \
                       (np.abs(z - soma_center[0]) <= axon_width // 2)

                # Create axon terminals
                axon_end = axon_start + axon_length
                terminals = np.zeros_like(soma, dtype=bool)
                for _ in range(3):  # Create 3 terminals
                    terminal_center = (
                        soma_center[0] + np.random.randint(-axon_width, axon_width),
                        soma_center[1] + np.random.randint(-axon_width, axon_width),
                        axon_end + np.random.randint(0, size[2]//10)
                    )
                    terminal_radius = axon_width // 2
                    dist_from_terminal = np.sqrt(
                        ((z - terminal_center[0]) * pixel_size[0])**2 +
                        ((y - terminal_center[1]) * pixel_size[1])**2 +
                        ((x - terminal_center[2]) * pixel_size[2])**2
                    )
                    terminals |= dist_from_terminal <= terminal_radius

                # Create dendrites
                dendrites = np.zeros_like(soma, dtype=bool)
                for _ in range(num_dendrites):
                    angle = np.random.uniform(0, 2*np.pi)
                    end_point = (
                        int(soma_center[0] + dendrite_length * np.sin(angle) * np.cos(angle)),
                        int(soma_center[1] + dendrite_length * np.sin(angle)),
                        int(soma_center[2] - dendrite_length * np.cos(angle))
                    )
                    rr, cc, zz = line_3d(soma_center[0], soma_center[1], soma_center[2],
                                         end_point[0], end_point[1], end_point[2])
                    dendrites[rr, cc, zz] = True

                dendrites = binary_dilation(dendrites, iterations=2)

                # Combine all parts to create the cell interior
                cell_interior = soma | axon | terminals | dendrites

                # Create the membrane by dilating the cell interior
                cell_shape = binary_dilation(cell_interior, iterations=membrane_thickness)
                cell_membrane = cell_shape & ~cell_interior

            elif cell_type == 'epithelial':
                height = kwargs.get('height', size[0] // 3)
                basal_membrane = np.zeros(size, dtype=bool)
                basal_membrane[0] = True

                apical_membrane = np.zeros(size, dtype=bool)
                apical_membrane[height-1] = True

                lateral_membrane = np.zeros(size, dtype=bool)
                lateral_membrane[:height, 0, :] = True
                lateral_membrane[:height, :, 0] = True
                lateral_membrane[:height, -1, :] = True
                lateral_membrane[:height, :, -1] = True

                cell_membrane = basal_membrane | apical_membrane | lateral_membrane
                cell_interior = np.zeros(size, dtype=bool)
                cell_interior[1:height-1, 1:-1, 1:-1] = True

                cell_shape = cell_membrane | cell_interior

            elif cell_type == 'muscle':
                # Ensure size is a tuple of integers
                z, y, x = [int(s[0] if isinstance(s, np.ndarray) else s) for s in size]
                size = (z, y, x)
                soma_center = np.array(size) // 2

                self.logger.info(f"Muscle cell size: z={z}, y={y}, x={x}")

                # Create a cylindrical shape for the muscle cell
                radius = min(y, z) // 4
                length = x

                # Create the main body of the muscle cell
                zz, yy, xx = np.ogrid[:z, :y, :x]
                cell_shape = (((yy - y//2)*pixel_size[1])**2 + ((zz - z//2)*pixel_size[0])**2 <= radius**2).astype(float)

                # Create a tapering function
                taper = np.linspace(0.7, 1, length//2)
                taper = np.concatenate([taper, taper[::-1]])

                # Apply tapering
                taper_3d = taper[np.newaxis, np.newaxis, :]
                cell_shape = cell_shape * taper_3d

                # Create the cell interior (slightly smaller than the full shape)
                cell_interior = (((yy - y//2)*pixel_size[1])**2 + ((zz - z//2)*pixel_size[0])**2 <= (radius - membrane_thickness)**2).astype(float)
                cell_interior = cell_interior * taper_3d

                # Create the cell membrane
                cell_membrane = (cell_shape > 0) & (cell_interior == 0)

                # Convert to float
                cell_shape = cell_shape.astype(float)
                cell_interior = cell_interior.astype(float)
                cell_membrane = cell_membrane.astype(float)


            else:
                raise ValueError(f"Unknown cell type: {cell_type}")

            non_zero_coords = np.argwhere(cell_shape > 0)
            self.logger.info(f"Non-zero cell shape coordinates: min={non_zero_coords.min(axis=0)}, max={non_zero_coords.max(axis=0)}")
            self.cell_shape = cell_shape.astype(float)
            self.logger.info(f"{cell_type} cell shape generated successfully")
            return cell_shape.astype(float), cell_interior.astype(float), cell_membrane.astype(float)

        except Exception as e:
            self.logger.error(f"Error in cell shape generation: {str(e)}")
            raise

    def _generate_nucleus(self) -> np.ndarray:
        # Implement nucleus generation here
        pass


    def generate_mitochondria(self, num_mitochondria: int = 50, size_range: Tuple[int, int] = (3, 8)):
        try:
            self.logger.info(f"Generating {num_mitochondria} mitochondria")

            if self.cell_shape is None or self.nucleus is None:
                raise ValueError("Cell shape and nucleus must be generated before mitochondria")

            # Convert float arrays to boolean
            cell_shape_bool = self.cell_shape > 0
            nucleus_bool = self.nucleus > 0
            cytoplasm = cell_shape_bool & ~nucleus_bool
            mitochondria = np.zeros_like(self.cell_shape, dtype=bool)

            # Get the actual size of the cell shape
            z_size, y_size, x_size = cytoplasm.shape

            generated_count = 0
            for _ in range(num_mitochondria):
                radius = np.random.uniform(size_range[0]/2, size_range[1]/2)
                pos = (np.random.randint(0, z_size),
                       np.random.randint(0, y_size),
                       np.random.randint(0, x_size))

                attempts = 0
                while not cytoplasm[pos] and attempts < 100:
                    pos = (np.random.randint(0, z_size),
                           np.random.randint(0, y_size),
                           np.random.randint(0, x_size))
                    attempts += 1

                if attempts == 100:
                    self.logger.warning("Failed to place mitochondrion after 100 attempts")
                    continue  # Skip this mitochondrion if we can't find a valid position

                z, y, x = np.ogrid[-int(radius):int(radius)+1, -int(radius):int(radius)+1, -int(radius):int(radius)+1]
                sphere = (x*x + y*y + z*z <= radius*radius)

                sz, sy, sx = sphere.shape
                pz, py, px = pos

                z_start, z_end = max(0, pz-int(radius)), min(z_size, pz+int(radius)+1)
                y_start, y_end = max(0, py-int(radius)), min(y_size, py+int(radius)+1)
                x_start, x_end = max(0, px-int(radius)), min(x_size, px+int(radius)+1)

                mito_slice = mitochondria[z_start:z_end, y_start:y_end, x_start:x_end]
                sphere_slice = sphere[:mito_slice.shape[0], :mito_slice.shape[1], :mito_slice.shape[2]]

                mito_slice |= sphere_slice
                mitochondria[z_start:z_end, y_start:y_end, x_start:x_end] = mito_slice

                generated_count += 1

            mitochondria &= cytoplasm
            self.mitochondria = mitochondria

            self.logger.info(f"Generated {generated_count} mitochondria. Total volume: {np.sum(mitochondria)}")
            return mitochondria.astype(float)

        except Exception as e:
            self.logger.error(f"Error in mitochondria generation: {str(e)}")
            raise


    def generate_cytoskeleton(self, actin_density: float = 0.05, microtubule_density: float = 0.02):
        try:
            self.logger.info("Generating cytoskeleton")

            if self.cell_shape is None or self.nucleus is None:
                raise ValueError("Cell shape and nucleus must be generated before cytoskeleton")

            # Convert float arrays to boolean
            cell_shape_bool = self.cell_shape > 0
            nucleus_bool = self.nucleus > 0
            cytoplasm = cell_shape_bool & ~nucleus_bool

            # Generate actin filaments
            actin = np.random.rand(*self.size) < actin_density
            actin &= cytoplasm
            actin = skeletonize(actin)

            # Generate microtubules
            microtubules = np.zeros_like(self.cell_shape, dtype=bool)
            center = np.array(self.nucleus.shape) // 2
            num_microtubules = int(microtubule_density * np.sum(cytoplasm))

            for _ in range(num_microtubules):
                end = np.array([np.random.randint(0, s) for s in self.size])
                rr, cc, vv = line_3d(*center, *end)

                # Clip coordinates to stay within bounds
                rr = np.clip(rr, 0, self.size[0] - 1)
                cc = np.clip(cc, 0, self.size[1] - 1)
                vv = np.clip(vv, 0, self.size[2] - 1)

                valid = cytoplasm[rr, cc, vv]
                microtubules[rr[valid], cc[valid], vv[valid]] = True

            microtubules = skeletonize(microtubules)
            microtubules &= cytoplasm  # Ensure microtubules don't enter the nucleus

            self.cytoskeleton = (actin, microtubules)

            self.logger.info(f"Generated cytoskeleton. Actin volume: {np.sum(actin)}, Microtubule volume: {np.sum(microtubules)}")
            return actin.astype(float), microtubules.astype(float)

        except Exception as e:
            self.logger.error(f"Error in cytoskeleton generation: {str(e)}")
            raise

    def simulate_active_transport(self, velocity: Tuple[float, float, float], cargo_concentration: np.ndarray, use_microtubules: bool = True) -> np.ndarray:
        try:
            if self.cytoskeleton is None:
                raise ValueError("Cytoskeleton must be generated before simulating active transport")

            actin, microtubules = self.cytoskeleton
            transport_network = microtubules if use_microtubules else actin

            self.logger.info(f"Starting active transport simulation. Transport network volume: {np.sum(transport_network)}")

            result = np.zeros((self.num_time_points, *self.size))
            # Place initial cargo on the transport network
            initial_cargo = cargo_concentration * transport_network
            if np.sum(initial_cargo) == 0:
                # If no cargo is on the network, place it at the nearest network point
                nearest_network_point = np.unravel_index(np.argmin(np.where(transport_network, 0, np.inf)), self.size)
                initial_cargo[nearest_network_point] = 1.0

            result[0] = initial_cargo
            self.logger.info(f"Initial cargo volume: {np.sum(result[0])}")

            for t in range(1, self.num_time_points):
                shifted = np.roll(result[t-1], shift=(int(velocity[0]), int(velocity[1]), int(velocity[2])), axis=(0,1,2))
                result[t] = shifted * transport_network

                # Add some diffusion to prevent cargo from disappearing
                result[t] = binary_dilation(result[t] > 0, iterations=1) * transport_network

                # Ensure cargo doesn't completely disappear
                if np.sum(result[t]) == 0:
                    result[t] = result[t-1]

                self.logger.info(f"Cargo volume at time {t}: {np.sum(result[t])}")

            self.logger.info(f"Final cargo volume: {np.sum(result[-1])}")
            return result
        except Exception as e:
            self.logger.error(f"Error in active transport simulation: {str(e)}")
            raise


    def _generate_actin(self) -> np.ndarray:
        # Implement actin network generation here
        pass

    def _generate_lysosomes(self) -> np.ndarray:
        # Implement lysosome generation here
        pass

    def simulate_calcium_signal(self, signal_type: str, params: Dict) -> np.ndarray:
        """
        Simulate calcium signaling events.

        Args:
        signal_type (str): Type of calcium signal ('blip', 'puff', 'wave')
        params (Dict): Parameters for the specific signal type

        Returns:
        np.ndarray: Time series of calcium concentrations
        """
        try:
            if signal_type == 'blip':
                return self._simulate_calcium_blip(params)
            elif signal_type == 'puff':
                return self._simulate_calcium_puff(params)
            elif signal_type == 'wave':
                return self._simulate_calcium_wave(params)
            else:
                raise ValueError(f"Unknown calcium signal type: {signal_type}")
        except Exception as e:
            self.logger.error(f"Error in calcium signal simulation: {str(e)}")
            raise

    def _simulate_calcium_blip(self, params: Dict) -> np.ndarray:
        """Simulate a calcium blip (localized, brief calcium release)"""
        duration = params.get('duration', 10)
        amplitude = params.get('amplitude', 1.0)
        location = params.get('location', tuple(s // 2 for s in self.size))

        result = np.zeros((self.num_time_points, *self.size))
        for t in range(min(duration, self.num_time_points)):
            result[t] = self._create_gaussian_spot(location, amplitude * (1 - t/duration))
        return result

    def _simulate_calcium_puff(self, params: Dict) -> np.ndarray:
        """Simulate a calcium puff (larger than a blip, involves multiple channels)"""
        duration = params.get('duration', 20)
        amplitude = params.get('amplitude', 2.0)
        location = params.get('location', tuple(s // 2 for s in self.size))
        spread = params.get('spread', 2.0)

        result = np.zeros((self.num_time_points, *self.size))
        for t in range(min(duration, self.num_time_points)):
            result[t] = self._create_gaussian_spot(location, amplitude * (1 - t/duration), spread)
        return result

    def _simulate_calcium_wave(self, params: Dict) -> np.ndarray:
        """Simulate a calcium wave (propagating elevation in calcium concentration)"""
        duration = params.get('duration', self.num_time_points)
        amplitude = params.get('amplitude', 3.0)
        speed = params.get('speed', 1.0)
        start_location = params.get('start_location', (0, self.size[1]//2, self.size[2]//2))

        result = np.zeros((self.num_time_points, *self.size))
        for t in range(min(duration, self.num_time_points)):
            wave_front = int(t * speed)
            wave_location = (start_location[0] + wave_front, start_location[1], start_location[2])
            result[t] = self._create_gaussian_spot(wave_location, amplitude, spread=5.0)
        return result

    def _create_gaussian_spot(self, center, amplitude, spread=1.0):
        z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
        r2 = ((z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2) / (2*spread**2)
        return amplitude * np.exp(-r2)

##########################################################################################
################## Shape Simulator   (for testing)      ##################################
##########################################################################################
class Shape(ABC):
    def __init__(self, position, size, color):
        self.position = np.array(position, dtype=float)
        self.size = size
        self.color = color

    @abstractmethod
    def get_vertices(self):
        pass

    @abstractmethod
    def get_faces(self):
        pass

    @abstractmethod
    def get_volume_points(self, resolution=10):
        pass

class Sphere(Shape):
    def get_volume_points(self, resolution=10):
        r = self.size / 2
        x, y, z = np.ogrid[-r:r:resolution*1j, -r:r:resolution*1j, -r:r:resolution*1j]
        mask = x**2 + y**2 + z**2 <= r**2
        points = np.column_stack(np.where(mask))
        return points + self.position

    def get_vertices(self):
        # Create a simple sphere approximation
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = self.size * np.outer(np.cos(u), np.sin(v))
        y = self.size * np.outer(np.sin(u), np.sin(v))
        z = self.size * np.outer(np.ones(np.size(u)), np.cos(v))
        vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
        return vertices + self.position

    def get_faces(self):
        # This is a simplified face generation and might not be perfect
        u = 20
        v = 10
        faces = []
        for i in range(u - 1):
            for j in range(v - 1):
                faces.append([i*v + j, (i+1)*v + j, i*v + (j+1)])
                faces.append([(i+1)*v + j, (i+1)*v + (j+1), i*v + (j+1)])
        return np.array(faces)

class Cube(Shape):
    def get_volume_points(self, resolution=10):
        s = self.size
        x, y, z = np.mgrid[0:s:resolution*1j, 0:s:resolution*1j, 0:s:resolution*1j]
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        return points + self.position - s/2

    def get_vertices(self):
        s = self.size / 2
        vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
        ])
        return vertices + self.position

    def get_faces(self):
        return np.array([
            [0, 1, 2], [0, 2, 3],  # front
            [1, 5, 6], [1, 6, 2],  # right
            [5, 4, 7], [5, 7, 6],  # back
            [4, 0, 3], [4, 3, 7],  # left
            [3, 2, 6], [3, 6, 7],  # top
            [4, 5, 1], [4, 1, 0],  # bottom
        ])

class Movement(ABC):
    @abstractmethod
    def update(self, shape, dt):
        pass

class RandomWalk(Movement):
    def __init__(self, speed):
        self.speed = speed

    def update(self, shape, dt):
        shape.position += np.random.normal(0, self.speed * dt, 3)

class LinearMotion(Movement):
    def __init__(self, velocity):
        self.velocity = np.array(velocity)

    def update(self, shape, dt):
        shape.position += self.velocity * dt

class Interaction(ABC):
    @abstractmethod
    def apply(self, shape1, shape2):
        pass

class Attraction(Interaction):
    def __init__(self, strength):
        self.strength = strength

    def apply(self, shape1, shape2):
        direction = shape2.position - shape1.position
        force = self.strength * direction / np.linalg.norm(direction)
        shape1.position += force
        shape2.position -= force

class Repulsion(Interaction):
    def __init__(self, strength, range):
        self.strength = strength
        self.range = range

    def apply(self, shape1, shape2):
        direction = shape2.position - shape1.position
        distance = np.linalg.norm(direction)
        if distance < self.range:
            force = self.strength * (self.range - distance) * direction / distance
            shape1.position -= force
            shape2.position += force

class ShapeSimulator:
    def __init__(self, size):
        self.size = size
        self.shapes = []
        self.movements = {}
        self.interactions = []

    def add_shape(self, shape, movement=None):
        self.shapes.append(shape)
        if movement:
            self.movements[shape] = movement

    def add_interaction(self, interaction):
        self.interactions.append(interaction)

    def update(self, dt):
        for shape, movement in self.movements.items():
            movement.update(shape, dt)

        for i, shape1 in enumerate(self.shapes):
            for shape2 in self.shapes[i+1:]:
                for interaction in self.interactions:
                    interaction.apply(shape1, shape2)

    def get_state(self, resolution=10):
        state = np.zeros(self.size, dtype=bool)
        for shape in self.shapes:
            points = shape.get_volume_points(resolution)
            indices = np.round(points).astype(int)
            valid_indices = np.all((indices >= 0) & (indices < np.array(self.size)), axis=1)
            valid_indices = indices[valid_indices]
            state[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True
        return state
