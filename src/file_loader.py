# file_loader.py
import os
import logging
import numpy as np
import tifffile
from PyQt5.QtWidgets import QMessageBox
from skimage import io
import nd2reader
import czifile
import json

# Set up logging
logger = logging.getLogger(__name__)

class FileLoader:
    def __init__(self):
        self.supported_extensions = {
            '.tif': self.load_tiff,
            '.tiff': self.load_tiff,
            '.nd2': self.load_nd2,
            '.lsm': self.load_tiff,  # LSM files are typically TIFF-based
            '.czi': self.load_czi,
            '.ome.tif': self.load_tiff,
            '.ome.tiff': self.load_tiff,
            '.jpg': self.load_general_image,
            '.png': self.load_general_image,
            '.bmp': self.load_general_image,
        }

    def load_file(self, filename):
        try:
            _, ext = os.path.splitext(filename.lower())
            if ext in self.supported_extensions:
                data, metadata = self.supported_extensions[ext](filename)
                return self.standardize_data(data, metadata)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            logger.error(f"Error loading file {filename}: {str(e)}")
            QMessageBox.critical(None, "Error", f"Failed to load file: {str(e)}")
            return None

    def load_tiff(self, filename):
        try:
            with tifffile.TiffFile(filename) as tif:
                data = tif.asarray()
                metadata = self.get_metadata_tiff(tif)
            return data, metadata
        except Exception as e:
            logger.error(f"Error loading TIFF file {filename}: {str(e)}")
            raise

    def load_nd2(self, filename):
        try:
            with nd2reader.ND2Reader(filename) as nd2:
                data = nd2.asarray()
                metadata = nd2.metadata
            return data, metadata
        except Exception as e:
            logger.error(f"Error loading ND2 file {filename}: {str(e)}")
            raise

    def load_czi(self, filename):
        try:
            with czifile.CziFile(filename) as czi:
                data = czi.asarray()
                metadata = czi.metadata()
            return np.squeeze(data), metadata  # Remove singleton dimensions
        except Exception as e:
            logger.error(f"Error loading CZI file {filename}: {str(e)}")
            raise

    def load_general_image(self, filename):
        try:
            data = io.imread(filename)
            metadata = {}
            return data, metadata
        except Exception as e:
            logger.error(f"Error loading image file {filename}: {str(e)}")
            raise

    def get_metadata_tiff(self, tif):
        metadata = {}
        try:
            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata is not None:
                metadata.update(tif.imagej_metadata)
            if hasattr(tif, 'metaseries_metadata') and tif.metaseries_metadata is not None:
                metadata.update(tif.metaseries_metadata)

            for page in tif.pages:
                if hasattr(page, 'tags'):
                    for tag in page.tags.values():
                        metadata[tag.name] = tag.value

            # Extract some common metadata
            metadata['shape'] = tif.series[0].shape if tif.series else None
            metadata['axes'] = tif.series[0].axes if tif.series else None
            metadata['dtype'] = str(tif.series[0].dtype) if tif.series else None

        except Exception as e:
            logger.warning(f"Error extracting metadata from TIFF: {str(e)}")

        return metadata

    def standardize_data(self, data, metadata):
        # Determine the number of dimensions and their meanings
        ndim = data.ndim
        shape = data.shape
        dims = metadata.get('axes', ['t', 'z', 'y', 'x', 'c'][-ndim:])

        # Create a standardized metadata dictionary
        std_metadata = {
            'dims': dims,
            'shape': shape,
            'dtype': str(data.dtype),
            'original_metadata': metadata
        }

        # Try to extract common metadata
        std_metadata['pixel_size_um'] = metadata.get('pixel_size_um', None)
        std_metadata['time_interval_s'] = metadata.get('time_interval_s', None)
        std_metadata['channel_names'] = metadata.get('channel_names', None)

        return data, std_metadata

    def save_metadata(self, filename, metadata):
        metadata_filename = os.path.splitext(filename)[0] + '_metadata.json'

        def json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.complex, np.complexfloating)):
                return str(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, bytes):
                return obj.decode('utf-8')
            raise TypeError(f"Type {type(obj)} not serializable")

        try:
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2, default=json_serializable)
            logger.info(f"Metadata saved to {metadata_filename}")
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_filename}: {str(e)}")
