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

    def add_points(self, points, window=None, additional_data=None):
        logger.info(f"Adding points. Shape of input: {points.shape if isinstance(points, pd.DataFrame) else len(points)}")

        if isinstance(points, pd.DataFrame):
            new_data = points.copy()
        else:
            if len(points[0]) == 5:  # Assuming [frame, x, y, z, t]
                new_data = pd.DataFrame(points, columns=['frame', 'x', 'y', 'z', 't'])
            else:
                raise ValueError(f"Expected 5 columns, got {len(points[0])}")

        # Add default columns if they don't exist
        if 'window' not in new_data.columns:
            new_data['window'] = window
        if 'selected' not in new_data.columns:
            new_data['selected'] = False
        if 'color' not in new_data.columns:
            new_data['color'] = '#FF0000'  # Default red color

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
        print(f"Points added. Total points: {len(self.data)}")


    def clear_points(self):
        self.data = pd.DataFrame()
        self.additional_columns = []
        self.data_changed.emit()

    def remove_points(self, point_ids):
        logger.info(f"Removing points with IDs: {point_ids}")
        self.data = self.data[~self.data.index.isin(point_ids)]
        self.data_changed.emit()
        logger.info(f"Points removed. Total points remaining: {len(self.data)}")

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

    # New methods for point selection and color change

    def select_points(self, point_ids):
        self.data.loc[self.data.index.isin(point_ids), 'selected'] = True
        self.data_changed.emit()

    def deselect_points(self, point_ids):
        self.data.loc[self.data.index.isin(point_ids), 'selected'] = False
        self.data_changed.emit()

    def change_points_color(self, point_ids, color):
        self.data.loc[self.data.index.isin(point_ids), 'color'] = color
        self.data_changed.emit()

    def get_points_in_window(self, window):
        return self.data[self.data['window'] == window]

    def get_points_in_roi(self, roi):
        mask = self.data.apply(lambda row: roi.contains(pg.Point(row['x'], row['y'])), axis=1)
        return self.data[mask]

    def plot_points(self, window, current_frame):
        logger.info(f"Plotting points for frame {current_frame}")

        # Clear all existing points
        if hasattr(window, 'point_items'):
            for item in window.point_items:
                window.get_view().removeItem(item)
            window.point_items.clear()
        else:
            window.point_items = []

        if self.data.empty or 'frame' not in self.data.columns:
            logger.info("No points data available")
            return 0

        frame_points = self.data[self.data['frame'] == current_frame]

        for _, point in frame_points.iterrows():
            point_item = pg.ScatterPlotItem([point['x']], [point['y']], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
            window.get_view().addItem(point_item)
            window.point_items.append(point_item)

        logger.info(f"Plotted {len(frame_points)} points")
        return len(frame_points)

    def get_points_in_frame(self, frame):
        if self.data.empty or 'frame' not in self.data.columns:
            return pd.DataFrame()
        return self.data[self.data['frame'] == frame]

