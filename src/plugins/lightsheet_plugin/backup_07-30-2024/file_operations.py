#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 07:31:46 2024

@author: george
"""

from abc import ABC, abstractmethod
import tifffile
import numpy as np

class MicroscopeDataImporter(ABC):
    @abstractmethod
    def read_metadata(self):
        pass

    @abstractmethod
    def read_data(self):
        pass

class ImporterFactory:
    @staticmethod
    def get_importer(file_path):
        if file_path.endswith('.tif') or file_path.endswith('.tiff'):
            return TIFFImporter(file_path)
        # Add more file formats here
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

# class SISImporter(MicroscopeDataImporter):
#     def __init__(self, file_path):
#         self.sis_reader = SISReader(file_path)

#     def read_metadata(self):
#         return self.sis_reader.read_metadata()

#     def read_data(self):
#         return self.sis_reader.read_data()

class TIFFImporter(MicroscopeDataImporter):
    def __init__(self, file_path):
        self.file_path = file_path

    def read_metadata(self):
        with tifffile.TiffFile(self.file_path) as tif:
            metadata = {}
            if hasattr(tif, 'imagej_metadata'):
                metadata.update(tif.imagej_metadata)
            if hasattr(tif, 'ome_metadata'):
                metadata.update(tif.ome_metadata)
            return metadata

    def read_data(self):
        with tifffile.TiffFile(self.file_path) as tif:
            data = tif.asarray()
            if data.ndim == 3:  # z, y, x
                data = data[np.newaxis, np.newaxis, :]  # Add t and c dimensions
            elif data.ndim == 4:  # c, z, y, x
                data = data[np.newaxis, :]  # Add t dimension
            return data
