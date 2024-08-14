#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:45:10 2024

@author: george
"""

# test_particle_detection.py

import unittest
import numpy as np
from skimage.draw import disk
from particle_analysis import ParticleAnalysisResults

class TestParticleDetection(unittest.TestCase):
    def generate_test_image(self, num_particles, image_size, particle_size, snr):
        image = np.random.normal(0, 1, (image_size, image_size))
        particles = []
        for _ in range(num_particles):
            x = np.random.randint(particle_size, image_size - particle_size)
            y = np.random.randint(particle_size, image_size - particle_size)
            rr, cc = disk((y, x), particle_size)
            image[rr, cc] += snr
            particles.append((y, x))
        return image, particles

    def test_particle_detection_varying_snr(self):
        image_size = 512
        particle_size = 5
        num_particles = 20
        snr_levels = [2, 5, 10, 20]

        for snr in snr_levels:
            with self.subTest(snr=snr):
                image, true_particles = self.generate_test_image(num_particles, image_size, particle_size, snr)

                analyzer = ParticleAnalysisResults(None, image)
                analyzer.options['min_area'] = np.pi * (particle_size - 1)**2
                analyzer.options['max_area'] = np.pi * (particle_size + 1)**2
                analyzer.options['threshold_method'] = 'otsu'
                analyzer.options['apply_threshold'] = True
                analyzer.options['apply_noise_reduction'] = True

                _, detected_particles = analyzer.process_frame(image)

                detected_coords = set(map(tuple, detected_particles[['centroid-0', 'centroid-1']].values))
                true_coords = set(map(tuple, true_particles))

                precision = len(detected_coords.intersection(true_coords)) / len(detected_coords) if detected_coords else 0
                recall = len(detected_coords.intersection(true_coords)) / len(true_coords) if true_coords else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f"SNR: {snr}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

                self.assertGreaterEqual(f1_score, 0.8, f"F1 score should be at least 0.8 for SNR {snr}")

if __name__ == '__main__':
    unittest.main()
