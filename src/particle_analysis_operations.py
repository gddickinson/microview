from PyQt5.QtWidgets import QPushButton, QMessageBox
from particle_analysis import ParticleAnalysisResults
import pyqtgraph as pg
import numpy as np

import logging

logger = logging.getLogger(__name__)

class ParticleAnalysisOperations:
    def __init__(self, parent):
        self.parent = parent
        self.toggle_centroids_button = QPushButton("Toggle Centroids")
        self.toggle_centroids_button.setCheckable(True)
        self.toggle_centroids_button.toggled.connect(self.toggle_centroids)
        self.toggle_centroids_button.setEnabled(False)
        self.centroids_visible = False

    def run_particle_analysis(self):
        if self.parent.window_management.current_window is None:
            QMessageBox.warning(self.parent, "No Image", "Please open an image first.")
            return

        try:
            image = self.parent.window_management.current_window.image
            analysis_dialog = ParticleAnalysisResults(self.parent, image)
            analysis_dialog.analysisComplete.connect(self.parent.on_particle_analysis_complete)
            analysis_dialog.exec_()
        except Exception as e:
            print(f"Error in particle analysis: {str(e)}")
            QMessageBox.critical(self.parent, "Error", f"Error in particle analysis: {str(e)}")

    def toggle_results_chart(self, checked):
        logger.info(f"Toggle results chart called with checked={checked}")
        if checked:
            self.show_results_chart()
        else:
            self.hide_results_chart()

    def show_results_chart(self):
        logger.info("Showing results chart")
        if not hasattr(self, 'results_chart_window'):
            self.results_chart_window = self.create_results_chart_window()
        self.results_chart_window.show()

    def hide_results_chart(self):
        logger.info("Hiding results chart")
        if hasattr(self, 'results_chart_window'):
            self.results_chart_window.hide()

    def create_results_chart_window(self):
        logger.info("Creating results chart window")
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableView
        from particle_analysis import PandasModel

        window = QWidget()
        layout = QVBoxLayout()

        # Create table view
        table_view = QTableView()
        model = PandasModel(self.parent.particle_analysis_results)
        table_view.setModel(model)
        layout.addWidget(table_view)

        # Create scatter plot
        scatter_widget = pg.PlotWidget()
        scatter_widget.setLabel('left', 'Y')
        scatter_widget.setLabel('bottom', 'X')
        scatter_widget.setTitle('Particle Locations and Trajectories')

        particles = self.parent.particle_analysis_results
        logger.info(f"Particles DataFrame shape: {particles.shape}")
        logger.info(f"Particles columns: {particles.columns.tolist()}")

        if 'x' in particles.columns and 'y' in particles.columns:
            x = particles['x'].values
            y = particles['y'].values
            scatter_plot = pg.ScatterPlotItem(x=x, y=y, size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
            scatter_widget.addItem(scatter_plot)
        else:
            logger.error("'x' or 'y' columns not found in particles DataFrame")

        layout.addWidget(scatter_widget)

        window.setLayout(layout)
        window.setWindowTitle("Particle Analysis Results")
        window.resize(800, 600)

        return window

    def toggle_centroids(self, checked):
        logger.info(f"Toggle centroids called with checked={checked}")
        self.centroids_visible = checked
        if self.parent.window_management.current_window:
            if checked:
                self.plot_centroids(self.parent.window_management.current_window)
            else:
                self.remove_centroids(self.parent.window_management.current_window)

    def plot_centroids(self, window):
        logger.info("Plotting centroids")
        current_frame = window.get_current_frame()
        self.plot_points(window, current_frame)

    def remove_centroids(self, window):
        logger.info("Removing centroids")
        if hasattr(window, 'point_items'):
            for item in window.point_items:
                window.get_view().removeItem(item)
            window.point_items.clear()

    def plot_points(self, window, current_frame):
        logger.info(f"Plotting points for frame {current_frame}")
        self.remove_centroids(window)

        frame_points = self.parent.point_data_manager.get_points_in_frame(current_frame)
        logger.info(f"Found {len(frame_points)} points for frame {current_frame}")
        logger.info(f"Columns in frame_points: {frame_points.columns.tolist()}")

        if frame_points.empty:
            logger.warning(f"No points found for frame {current_frame}")
            return

        for _, point in frame_points.iterrows():
            point_item = pg.ScatterPlotItem([point['x']], [point['y']], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
            window.get_view().addItem(point_item)
            if not hasattr(window, 'point_items'):
                window.point_items = []
            window.point_items.append(point_item)

        logger.info(f"Plotted {len(window.point_items)} points on the image")

        if hasattr(self.parent, 'particle_count_label'):
            particle_count = len(frame_points)
            self.parent.particle_count_label.setText(f"Particles in frame: {particle_count}")

    def update_point_data_manager(self, df):
        logger.info("Updating point data manager")
        self.parent.point_data_manager.clear_points()

        # Ensure all required columns are present
        required_columns = ['frame', 'x', 'y', 'z', 't']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Adding it with default values.")
                if col == 'frame' or col == 't':
                    df[col] = df.index
                else:
                    df[col] = 0

        logger.info(f"Columns in df before adding points: {df.columns.tolist()}")
        self.parent.point_data_manager.add_points(
            df[required_columns].values,
            additional_data={col: df[col].values for col in df.columns if col not in required_columns}
        )
        logger.info(f"Added {len(df)} points to point data manager")
        logger.info(f"Columns in point data manager: {self.parent.point_data_manager.data.columns.tolist()}")


    def on_time_slider_changed(self):
        logger.info("Time slider changed")
        if self.centroids_visible:
            current_window = self.parent.window_management.current_window
            if current_window:
                self.plot_centroids(current_window)
