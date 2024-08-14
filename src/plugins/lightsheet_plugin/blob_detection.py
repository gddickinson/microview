#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 07:05:28 2024

@author: george
"""

import numpy as np
import logging


from scipy.spatial import cKDTree


class BlobAnalyzer:
    def __init__(self, blobs):
        self.blobs = blobs
        self.channels = np.unique(blobs[:, 4]).astype(int)
        self.time_points = np.unique(blobs[:, 5]).astype(int)

    def calculate_nearest_neighbor_distances(self, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]

        # All blobs
        all_distances = self._calculate_nn_distances(time_blobs[:, :3])

        # Within channels
        within_channel_distances = {ch: self._calculate_nn_distances(time_blobs[time_blobs[:, 4] == ch][:, :3])
                                    for ch in self.channels}

        # Between channels
        between_channel_distances = {}
        for ch1 in self.channels:
            for ch2 in self.channels:
                if ch1 < ch2:
                    blobs1 = time_blobs[time_blobs[:, 4] == ch1][:, :3]
                    blobs2 = time_blobs[time_blobs[:, 4] == ch2][:, :3]
                    between_channel_distances[(ch1, ch2)] = self._calculate_cross_distances(blobs1, blobs2)

        return all_distances, within_channel_distances, between_channel_distances

    def _calculate_nn_distances(self, points):
        if len(points) < 2:
            return np.array([])
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        return distances[:, 1]  # Exclude self-distance

    def _calculate_cross_distances(self, points1, points2):
        if len(points1) == 0 or len(points2) == 0:
            return np.array([])
        tree = cKDTree(points2)
        distances, _ = tree.query(points1)
        return distances

    def calculate_blob_density(self, volume_size, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        total_volume = np.prod(volume_size)
        channel_densities = {ch: np.sum(time_blobs[:, 4] == ch) / total_volume for ch in self.channels}
        overall_density = len(time_blobs) / total_volume
        return overall_density, channel_densities

    def calculate_colocalization(self, distance_threshold, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        colocalization = {}
        for ch1 in self.channels:
            for ch2 in self.channels:
                if ch1 < ch2:
                    blobs1 = time_blobs[time_blobs[:, 4] == ch1][:, :3]
                    blobs2 = time_blobs[time_blobs[:, 4] == ch2][:, :3]
                    if len(blobs1) > 0 and len(blobs2) > 0:
                        tree = cKDTree(blobs2)
                        distances, _ = tree.query(blobs1)
                        colocalization[(ch1, ch2)] = np.mean(distances < distance_threshold)
                    else:
                        colocalization[(ch1, ch2)] = 0
        return colocalization

    def calculate_blob_sizes(self, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        return {ch: time_blobs[time_blobs[:, 4] == ch][:, 3] for ch in self.channels}

    def calculate_blob_intensities(self, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        return {ch: time_blobs[time_blobs[:, 4] == ch][:, 6] for ch in self.channels}  # Column 6 is now intensity

    def calculate_advanced_colocalization(self, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        results = {}
        for ch1 in self.channels:
            for ch2 in self.channels:
                if ch1 < ch2:
                    blobs1 = time_blobs[time_blobs[:, 4] == ch1]
                    blobs2 = time_blobs[time_blobs[:, 4] == ch2]
                    if len(blobs1) > 0 and len(blobs2) > 0:
                        # Use KDTree for efficient nearest neighbor search
                        tree1 = cKDTree(blobs1[:, :3])  # x, y, z coordinates
                        tree2 = cKDTree(blobs2[:, :3])

                        # Calculate distance threshold based on mean blob size
                        distance_threshold = np.mean(np.concatenate([blobs1[:, 3], blobs2[:, 3]]))

                        # Find nearest neighbors
                        distances, indices = tree1.query(blobs2[:, :3])

                        # Calculate intensity correlations for nearby blobs
                        nearby_mask = distances < distance_threshold
                        intensities1 = blobs1[indices[nearby_mask], 6]
                        intensities2 = blobs2[nearby_mask, 6]

                        if len(intensities1) > 1 and len(intensities2) > 1:
                            pearson = np.corrcoef(intensities1, intensities2)[0, 1]
                        else:
                            pearson = np.nan

                        # Calculate Manders' coefficients
                        m1 = np.sum(intensities1) / np.sum(blobs1[:, 6])
                        m2 = np.sum(intensities2) / np.sum(blobs2[:, 6])

                        results[(ch1, ch2)] = {'pearson': pearson, 'manders_m1': m1, 'manders_m2': m2}
                    else:
                        results[(ch1, ch2)] = {'pearson': np.nan, 'manders_m1': np.nan, 'manders_m2': np.nan}
        return results
