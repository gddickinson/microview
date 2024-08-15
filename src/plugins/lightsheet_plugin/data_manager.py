#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:27:19 2024

@author: george
"""


import numpy as np
import logging
from data_generation import DataGenerator
from lightsheet_file_operations import ImporterFactory
from data_properties_dialog import DataPropertiesDialog
from PyQt5.QtWidgets import QApplication

class DataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        #self.data = None
        self.raw_data = None
        self.processed_data = None
        self.metadata = {}
        self.data_properties = None
        self.data_generator = DataGenerator()

    def generate_data(self, params):
        try:
            if params['structured_data']:
                self.processed_data = self.generate_structured_data(params)
            else:
                self.processed_data = self.generate_unstructured_data(params)

            # Set raw_data as a reshaped version of processed_data
            self.raw_data = self.processed_data.reshape(-1, *self.processed_data.shape[-2:])

            self.data_properties = {
                'num_z_slices': self.processed_data.shape[2],
                'num_channels': self.processed_data.shape[1],
                'pixel_size_xy': 1.0,
                'pixel_size_z': 1.0,
                'slice_angle': 90
            }

            self.log_data_info()
            return self.processed_data
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
        if self.raw_data is not None:
            self.logger.info(f"Raw data shape: {self.raw_data.shape}")
        if self.processed_data is not None:
            self.logger.info(f"Processed data shape: {self.processed_data.shape}")
        self.logger.info(f"Data properties: {self.data_properties}")
        if self.processed_data is not None:
            self.logger.info(f"Data min: {self.processed_data.min()}, max: {self.processed_data.max()}, mean: {self.processed_data.mean()}")


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
            self.raw_data = importer.read_data()

            if self.raw_data is None:
                raise ValueError("Failed to read data from the file.")

            # Show data properties dialog
            self.get_data_properties()

            # Process the raw data
            self.process_raw_data()

            self.log_data_info()
            return self.processed_data, self.metadata, self.data_properties
        except Exception as e:
            self.logger.error(f"Error importing microscope data: {str(e)}")
            raise

    def get_data_properties(self):
        dialog = DataPropertiesDialog()
        if dialog.exec_():
            self.data_properties = dialog.get_properties()
        else:
            raise ValueError("Data properties not set. Cannot proceed with import.")

    def process_raw_data(self):
        if self.raw_data is None or self.data_properties is None:
            raise ValueError("Raw data or data properties not available.")

        num_slices = self.data_properties['num_z_slices']
        num_channels = self.data_properties['num_channels']

        # Assuming raw_data is a 3D array (frames, height, width)
        total_frames = self.raw_data.shape[0]
        num_timepoints = total_frames // (num_slices * num_channels)

        # Reshape the data to (t, c, z, y, x)
        self.processed_data = self.raw_data.reshape(num_timepoints, num_channels, num_slices,
                                                    self.raw_data.shape[1], self.raw_data.shape[2])

    # def log_data_info(self):
    #     self.logger.info(f"Imported raw data shape: {self.raw_data.shape}")
    #     self.logger.info(f"Processed data shape: {self.processed_data.shape}")
    #     self.logger.info(f"Data properties: {self.data_properties}")
    #     self.logger.info(f"Data min: {self.processed_data.min()}, max: {self.processed_data.max()}, mean: {self.processed_data.mean()}")


    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.metadata
