import pandas as pd
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import logging
import pyqtgraph as pg

logger = logging.getLogger(__name__)

class PointDataManager(QObject):
    data_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = pd.DataFrame()
        self.additional_columns = []

    def add_points(self, points, additional_data=None):
        logger.info(f"Adding points. Shape of input: {points.shape}")

        if isinstance(points, pd.DataFrame):
            new_data = points
        else:
            if points.shape[1] == 3:
                new_data = pd.DataFrame(points, columns=['frame', 'x', 'y'])
            elif points.shape[1] == 4:
                new_data = pd.DataFrame(points, columns=['frame', 'x', 'y', 'z'])
            elif points.shape[1] >= 5:
                new_data = pd.DataFrame(points, columns=['frame', 'x', 'y', 'z', 't'])
            else:
                raise ValueError(f"Expected at least 3 columns, got {points.shape[1]}")

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

    def get_points_in_frame(self, frame):
        return self.data[self.data['frame'] == frame]

    def clear_points(self):
        self.data = pd.DataFrame()
        self.additional_columns = []
        self.data_changed.emit()

    def remove_points(self, point_ids):
        self.data = self.data[~self.data.index.isin(point_ids)]
        self.data_changed.emit()

    def link_points(self, point_ids):
        if 'particle' not in self.data.columns:
            self.data['particle'] = np.nan
        new_particle_id = self.data['particle'].max() + 1 if not pd.isna(self.data['particle'].max()) else 0
        self.data.loc[self.data.index.isin(point_ids), 'particle'] = new_particle_id
        self.data_changed.emit()

    def move_point(self, point_id, new_x, new_y):
        self.data.loc[point_id, 'x'] = new_x
        self.data.loc[point_id, 'y'] = new_y
        self.data_changed.emit()

    def remove_points_in_roi(self, roi):
        mask = self.data.apply(lambda row: roi.contains(pg.Point(row['x'], row['y'])), axis=1)
        self.data = self.data[~mask]
        self.data_changed.emit()

    def move_points_in_roi(self, roi, dx, dy):
        mask = self.data.apply(lambda row: roi.contains(pg.Point(row['x'], row['y'])), axis=1)
        self.data.loc[mask, 'x'] += dx
        self.data.loc[mask, 'y'] += dy
        self.data_changed.emit()

    def update_time_values(self, time_interval):
        """
        Update 't' values based on a given time interval
        :param time_interval: time between frames in seconds
        """
        self.data['t'] = self.data['frame'] * time_interval
        self.data_changed.emit()

    def get_data(self):
        return self.data

    def set_data(self, new_data):
        self.data = new_data
        self.data_changed.emit()
