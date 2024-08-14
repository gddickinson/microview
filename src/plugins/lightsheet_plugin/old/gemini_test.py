import numpy as np
import logging

class DataGenerator:
    def __init__(self):
        self.data = None
        self.logger = logging.getLogger(__name__)

    def generate_random_noise(self, size=(100, 100, 30), noise_level=0.1):
        """Generates a 3D volume of random noise."""
        try:
            self.data = np.random.rand(*size) * noise_level
            self.logger.info(f"Generated random noise data with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            self.logger.error(f"Error generating random noise: {e}")
            raise  # Re-raise the exception for higher-level handling


