#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:27:19 2024

@author: george
"""


import numpy as np
import logging
from data_generation import DataGenerator
from file_operations import ImporterFactory

class DataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data = None
        self.metadata = {}
        self.data_generator = DataGenerator()

    def generate_data(self, params):
        try:
            if params['structured_data']:
                self.data = self.generate_structured_data(params)
            else:
                self.data = self.generate_unstructured_data(params)

            self.log_data_info()
            return self.data
        except Exception as e:
            self.logger.error(f"Error in data generation: {str(e)}")
            raise

    def generate_structured_data(self, params):
        channel_ranges = [
            ((ch['x'][0], ch['x'][1]), (ch['y'][0], ch['y'][1]), (ch['z'][0], ch['z'][1]))
            for ch in params['channel_ranges']
        ]
        return self.data_generator.generate_structured_multi_channel_time_series(
            num_volumes=params['num_volumes'],
            num_channels=params['num_channels'],
            size=params['size'],
            num_blobs=params['num_blobs'],
            intensity_range=(0.8, 1.0),
            sigma_range=(2, 6),
            noise_level=params['noise_level'],
            movement_speed=params['movement_speed'],
            channel_ranges=channel_ranges
        )

    def generate_unstructured_data(self, params):
        return self.data_generator.generate_multi_channel_time_series(
            num_volumes=params['num_volumes'],
            num_channels=params['num_channels'],
            size=params['size'],
            num_blobs=params['num_blobs'],
            intensity_range=(0.8, 1.0),
            sigma_range=(2, 6),
            noise_level=params['noise_level'],
            movement_speed=params['movement_speed']
        )

    def log_data_info(self):
        self.logger.info(f"Generated data shape: {self.data.shape}")
        self.logger.info(f"Data min: {self.data.min()}, max: {self.data.max()}, mean: {self.data.mean()}")

    def load_data(self, filename):
        try:
            if filename.endswith('.tiff'):
                self.data = self.data_generator.load_tiff(filename)
            elif filename.endswith('.npy'):
                self.data = self.data_generator.load_numpy(filename)
            else:
                raise ValueError("Unsupported file format")

            self.log_data_info()
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def save_data(self, filename):
        try:
            if filename.endswith('.tiff'):
                self.data_generator.save_tiff(filename)
            elif filename.endswith('.npy'):
                self.data_generator.save_numpy(filename)
            else:
                raise ValueError("Unsupported file format")

            self.logger.info(f"Data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def import_microscope_data(self, file_path):
        try:
            importer = ImporterFactory.get_importer(file_path)
            self.metadata = importer.read_metadata()
            self.data = importer.read_data()

            if self.data is None:
                raise ValueError("Failed to read data from the file.")

            self.log_data_info()
            return self.data, self.metadata
        except Exception as e:
            self.logger.error(f"Error importing microscope data: {str(e)}")
            raise

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.metadata
