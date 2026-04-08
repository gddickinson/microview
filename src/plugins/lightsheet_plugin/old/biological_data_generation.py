#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:15:50 2024

@author: george
"""

# biological_data_generation.py


import numpy as np
from scipy.ndimage import gaussian_filter
from base_data_generator import DataGenerator

class BiologicalDataGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.structures = {}

    def generate_volume(self, params):
        size = params.get('size', (100, 100, 30))
        cell_radius = params.get('cell_radius', min(size) // 4)

        self.generate_cell_structures(size, cell_radius)

        volume = np.zeros(size)
        for structure, (data, intensity) in self.structures.items():
            volume[data] = intensity

        return volume

    def generate_cell_structures(self, size, cell_radius):
        z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
        center = np.array(size) // 2
        dist_from_center = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)

        # Membrane
        membrane = np.logical_and(dist_from_center >= cell_radius - 1, dist_from_center <= cell_radius)
        self.structures['membrane'] = (membrane, 0.8)

        # Nucleus
        nucleus_radius = cell_radius // 2
        nucleus = dist_from_center <= nucleus_radius
        self.structures['nucleus'] = (nucleus, 0.6)

        # Cytoplasm
        cytoplasm = np.logical_and(dist_from_center < cell_radius, dist_from_center > nucleus_radius)
        self.structures['cytoplasm'] = (cytoplasm, 0.4)

        # Endoplasmic Reticulum (ER)
        er = self.generate_er(size, nucleus_radius, cell_radius)
        self.structures['er'] = (er, 0.5)

        # Mitochondria
        mitochondria = self.generate_mitochondria(size, nucleus_radius, cell_radius)
        self.structures['mitochondria'] = (mitochondria, 0.7)

    def generate_er(self, size, nucleus_radius, cell_radius):
        z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
        center = np.array(size) // 2
        dist_from_center = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)

        er_mask = np.logical_and(dist_from_center > nucleus_radius, dist_from_center < cell_radius)
        er = np.random.rand(*size) < 0.3  # 30% chance of ER at each point
        return np.logical_and(er, er_mask)

    def generate_mitochondria(self, size, nucleus_radius, cell_radius):
        mitochondria = np.zeros(size, dtype=bool)
        num_mitochondria = 20
        for _ in range(num_mitochondria):
            pos = np.random.rand(3) * (cell_radius - nucleus_radius) + nucleus_radius
            pos = pos.astype(int)
            mitochondria[pos[0]-2:pos[0]+2, pos[1]-2:pos[1]+2, pos[2]-2:pos[2]+2] = True
        return mitochondria

    def generate_time_series(self, params):
        num_volumes = params.get('num_volumes', 10)
        size = params.get('size', (100, 100, 30))

        time_series = np.zeros((num_volumes, *size))
        proteins = self.initialize_proteins(size)

        for t in range(num_volumes):
            volume = self.generate_volume(params)
            proteins = self.diffuse_proteins(proteins)
            volume += proteins
            time_series[t] = volume

        return time_series

    def initialize_proteins(self, size):
        return np.random.rand(*size) * 0.2  # Initialize proteins with low concentration

    def diffuse_proteins(self, proteins, sigma=0.5):
        return gaussian_filter(proteins, sigma=sigma)
