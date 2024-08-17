import pandas as pd
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import logging

logger = logging.getLogger(__name__)


class PointDataManager(QObject):
    data_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = pd.DataFrame(columns=['frame', 'x', 'y', 'z', 't'])
        self.additional_columns = []

    def add_points(self, points, additional_data=None):
        logger.info(f"Adding points. Shape of input: {points.shape}")
        logger.info(f"Columns in input: {points.shape[1]}")

        if points.shape[1] == 3:
            new_data = pd.DataFrame(points, columns=['frame', 'x', 'y'])
            new_data['z'] = 0  # Add z coordinate with default value 0
            new_data['t'] = new_data['frame']  # Assume t is same as frame if not provided
        elif points.shape[1] == 4:
            new_data = pd.DataFrame(points, columns=['frame', 'x', 'y', 'z'])
            new_data['t'] = new_data['frame']  # Assume t is same as frame if not provided
        elif points.shape[1] == 5:
            new_data = pd.DataFrame(points, columns=['frame', 'x', 'y', 'z', 't'])
        else:
            raise ValueError(f"Expected 3, 4 or 5 columns, got {points.shape[1]}")

        logger.info(f"Columns in new_data: {new_data.columns.tolist()}")

        if additional_data:
            for col, values in additional_data.items():
                new_data[col] = values
                if col not in self.additional_columns:
                    self.additional_columns.append(col)

        self.data = pd.concat([self.data, new_data], ignore_index=True)
        logger.info(f"Data after addition: {self.data.shape}")
        logger.info(f"Columns in data: {self.data.columns.tolist()}")
        self.data_changed.emit()

    def remove_points(self, indices):
        """Remove points at the specified indices."""
        self.data = self.data.drop(indices).reset_index(drop=True)
        self.data_changed.emit()

    def clear_points(self):
        """Clear all points."""
        self.data = pd.DataFrame(columns=['frame', 'x', 'y', 'z', 't'])
        self.additional_columns = []
        self.data_changed.emit()

    def get_points(self, dimensions=None, additional_columns=None):
        """
        Get points, optionally filtering dimensions and additional columns.

        :param dimensions: List of dimensions to include (e.g., ['x', 'y', 'z'])
        :param additional_columns: List of additional columns to include
        :return: DataFrame with requested data
        """
        if dimensions is None:
            dimensions = ['x', 'y', 'z', 't']

        columns = dimensions + (additional_columns or [])
        return self.data[columns]

    def save_points(self, filename):
        """Save points to a CSV file."""
        self.data.to_csv(filename, index=False)

    def load_points(self, filename):
        """Load points from a CSV file."""
        self.data = pd.read_csv(filename)
        self.additional_columns = [col for col in self.data.columns if col not in ['x', 'y', 'z', 't']]
        self.data_changed.emit()

    def update_point(self, index, new_data):
        """Update a single point with new data."""
        for key, value in new_data.items():
            self.data.at[index, key] = value
        self.data_changed.emit()

    def get_points_in_frame(self, frame):
        """Get points in a specific time frame."""
        if 'frame' not in self.data.columns:
            logger.error("'frame' column not found in data")
            return pd.DataFrame()
        return self.data[self.data['frame'] == frame]

    def get_points_in_roi(self, roi):
        """Get points within a specific ROI."""
        # Implement ROI filtering logic here
        # This will depend on how ROIs are defined in MicroView
        pass

    def add_column(self, column_name, default_value=None):
        """Add a new column to the data."""
        if column_name not in self.data.columns:
            self.data[column_name] = default_value
            self.additional_columns.append(column_name)
            self.data_changed.emit()

    def remove_column(self, column_name):
        """Remove a column from the data."""
        if column_name in self.additional_columns:
            self.data = self.data.drop(columns=[column_name])
            self.additional_columns.remove(column_name)
            self.data_changed.emit()

    def update_time_values(self, time_interval):
        """
        Update 't' values based on a given time interval
        :param time_interval: time between frames in seconds
        """
        self.data['t'] = self.data['frame'] * time_interval
        self.data_changed.emit()
