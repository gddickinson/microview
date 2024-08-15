# microview.py

import sys
import logging
from logging_config import setup_logging
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                             QDockWidget, QListWidget, QPushButton, QVBoxLayout,
                             QWidget, QToolBar, QMenuBar, QMenu, QInputDialog,
                             QColorDialog, QMessageBox, QComboBox, QTableView,
                             QLabel, QSplitter, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import numpy as np
import tifffile
import importlib.util
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation


from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from IPython import get_ipython

import os
import json
from logging.handlers import RotatingFileHandler
from importlib.metadata import version
import importlib

from filters import Filters
from particle_analysis import PandasModel, ParticleAnalysisResults
from menu_manager import MenuManager
from image_processing import ImageProcessor
from window_manager import WindowManager
from roi import RectROI, EllipseROI, LineROI

import pandas as pd
from scipy import stats

from flika_compatibility import flika_open_file, flika_open_rois, FlikaMicroViewWindow, FLIKA_AVAILABLE

import importlib.util
from info_panel import InfoPanel
from image_window import ImageWindow
from z_profile import ZProfileWidget
from roi_info import ROIInfoWidget
from roi_z_profile import ROIZProfileWidget
from file_loader import FileLoader
from scikit_analysis_console import ScikitAnalysisConsole
from transformations_dialog import TransformationsDialog
from synthetic_data_dialog import SyntheticDataDialog
from math_operations import MathOperations
from filter_operations import FilterOperations
from binary_operations import BinaryOperations
from stack_operations import StackOperations
from analysis_operations import AnalysisOperations
from roi_operations import ROIOperations
from particle_analysis_operations import ParticleAnalysisOperations
from ui_operations import UIOperations
from plugin_management import PluginManagement
from file_operations import FileOperations
from logging_operations import LoggingOperations
from config_management import ConfigManagement
from window_management_operations import WindowManagementOperations
from variable_management import VariableManagement
from console_operations import ConsoleOperations
from menu_operations import MenuOperations
from metadata_operations import MetadataOperations


#setup loggin
logger = setup_logging()

class MicroViewConsole(RichJupyterWidget):
    def __init__(self, parent=None):
        super(MicroViewConsole, self).__init__(parent)
        self._create_new_kernel()

    def _create_new_kernel(self):
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
        self.exit_requested.connect(stop)

    def push_variables(self, variables):
        self.kernel_manager.kernel.shell.push(variables)


class MicroView(QMainWindow):
    VERSION = "1.0.0"

    def __init__(self):
        super().__init__()
        self.logger = LoggingOperations.setup_logging()
        self.config_manager = ConfigManagement(self)
        self.config_manager.load_config()
        self.ui_operations = UIOperations(self)
        self.plugin_management = PluginManagement(self)
        self.window_management = WindowManagementOperations(self)
        self.file_operations = FileOperations(self)
        self.console_operations = ConsoleOperations(self)
        self.variable_management = VariableManagement(self)
        self.menu_operations = MenuOperations(self)
        self.metadata_operations = MetadataOperations(self)
        self.in_spyder = get_ipython().__class__.__name__ == 'SpyderShell'
        self.filters = Filters(self)
        self.particle_analysis_results = None
        self.menu_manager = MenuManager(self)
        self.image_processor = ImageProcessor()
        self.file_loader = FileLoader()
        self.initUI()
        self.plugins = self.plugin_management.load_plugins()
        self.logger.info(f"Plugins loaded: {list(self.plugins.keys())}")
        self.plugin_management.update_plugin_list(self.pluginList)
        self.setupMenus()
        self.loadBuiltInOperations()

    def initUI(self):
        self.setWindowTitle('MicroView Control Panel')
        self.setGeometry(100, 100, 1200, 800)

        # Create toolbar
        self.toolbar, self.toggle_chart_button, self.toggle_centroids_button = self.ui_operations.createToolbar()
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # Create dock widgets
        self.info_dock, self.info_panel = self.ui_operations.createInfoDock()
        self.z_profile_dock, self.z_profile_widget = self.ui_operations.createZProfileDock()
        self.plugin_dock, self.pluginList, self.runPluginButton = self.ui_operations.createPluginDock()
        self.roi_tools_dock, self.roi_info_widget, self.roi_z_profile_widget, self.roi_zoom_view = self.ui_operations.createROIToolsDock()

        # Create console dock
        self.console_operations.create_console_dock()

        # Arrange docks
        self.addDockWidget(Qt.TopDockWidgetArea, self.info_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.z_profile_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.roi_tools_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plugin_dock)
        if self.console_operations.console_dock:
            self.addDockWidget(Qt.BottomDockWidgetArea, self.console_operations.console_dock)

        # Hide some docks by default
        self.roi_tools_dock.hide()
        self.plugin_dock.hide()

        # Particle count label in status bar
        self.particle_count_label = self.ui_operations.createParticleCountLabel()
        self.statusBar().addPermanentWidget(self.particle_count_label)

        # Connect signals
        self.window_management.current_window_changed.connect(self.on_current_window_changed)


        # Initialize shared variables
        self.variable_management.push_variables({
            'np': np,
            'pg': pg,
            'filters': filters,
            'morphology': morphology,
            'measure': measure,
            'segmentation': segmentation,
            'g': g
        })

        self.show()

    def on_current_window_changed(self, window):
        if hasattr(self, 'info_panel'):
            self.info_panel.update_info(window)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.ui_operations.adjustDockSizes(self.width(), self.height())

    def loadBuiltInOperations(self):
        self.math_operations = MathOperations(self)
        self.filter_operations = FilterOperations(self)
        self.binary_operations = BinaryOperations(self)
        self.stack_operations = StackOperations(self)
        self.analysis_operations = AnalysisOperations(self)
        self.roi_operations = ROIOperations(self)
        self.particle_analysis_operations = ParticleAnalysisOperations(self)

        logger.info("Built-in operations loaded")

    def mathOperation(self, operation):
        self.math_operations.mathOperation(operation)

    def gaussianBlur(self):
        self.filter_operations.gaussianBlur()

    def medianFilter(self):
        self.filter_operations.medianFilter()

    def sobelEdge(self):
        self.filter_operations.sobelEdge()

    def threshold(self):
        self.binary_operations.threshold()

    def erode(self):
        self.binary_operations.erode()

    def dilate(self):
        self.binary_operations.dilate()

    def zProjectMax(self):
        self.stack_operations.zProjectMax()

    def zProjectMean(self):
        self.stack_operations.zProjectMean()

    def measure(self):
        self.analysis_operations.measure()

    def findMaxima(self):
        self.analysis_operations.findMaxima()

    def run_particle_analysis(self):
        self.particle_analysis_operations.run_particle_analysis()

    def toggle_results_chart(self, checked):
        self.particle_analysis_operations.toggle_results_chart(checked)

    def toggle_centroids(self, checked):
        self.particle_analysis_operations.toggle_centroids(checked)


    def colocalization_analysis(self):
        self.analysis_operations.colocalization_analysis()

    def open_analysis_console(self):
        self.analysis_operations.open_analysis_console()

    def addROI(self, roi_type):
        self.roi_operations.addROI(roi_type)

    def removeAllROIs(self):
        self.roi_operations.removeAllROIs()

    def save_rois_dialog(self):
        self.roi_operations.save_rois_dialog()

    def load_rois_dialog(self):
        self.roi_operations.load_rois_dialog()

    def openFile(self):
        self.file_operations.open_file()

    def loadImage(self, fileName):
        self.file_operations.load_image(fileName)

    def saveFile(self):
        self.file_operations.save_file()

    def load_plugins(self):
        self.plugin_management.load_plugins()

    def update_plugin_list(self):
        self.plugin_management.update_plugin_list(self.pluginList)

    def runSelectedPlugin(self):
        self.plugin_management.run_selected_plugin(self.pluginList)

    def close_all_plugins(self):
        self.plugin_management.close_all_plugins()

    def save_config(self):
        self.config_manager.save_config()

    def set_current_window(self, window):
        self.window_management.set_current_window(window)

    def add_window(self, window):
        return self.window_management.add_window(window)

    def closeCurrentWindow(self):
        self.window_management.close_current_window()

    def tileWindows(self):
        self.window_management.tile_windows()

    def cascadeWindows(self):
        self.window_management.cascade_windows()

    def push_variables(self, variables):
        self.variable_management.push_variables(variables)

    def get_variable(self, name):
        return self.variable_management.get_variable(name)

    def toggleConsole(self):
        self.console_operations.toggle_console()

    def show_ipython_commands(self):
        self.console_operations.show_ipython_commands()

    def add_menu_item(self, menu_name, item_name, callback):
        self.menu_operations.add_menu_item(menu_name, item_name, callback)

    def show_user_guide(self):
        self.menu_operations.show_user_guide()

    def show_about(self):
        self.menu_operations.show_about()

    def display_metadata_info(self, metadata):
        self.metadata_operations.display_metadata_info(metadata)

    def update_recent_files_menu(self):
        self.menu_manager.update_recent_files_menu()

    def on_time_slider_changed(self):
        if self.window_management.current_window:
            self.window_management.current_window.update_frame_info()


    def setupPluginDock(self):
        self.plugin_dock= QDockWidget("Plugins", self)
        pluginWidget = QWidget()
        pluginLayout = QVBoxLayout(pluginWidget)
        self.pluginList = QListWidget()
        self.pluginList.itemDoubleClicked.connect(self.runSelectedPlugin)
        pluginLayout.addWidget(self.pluginList)
        self.runPluginButton = QPushButton("Run Selected Plugin")
        self.runPluginButton.clicked.connect(self.runSelectedPlugin)
        pluginLayout.addWidget(self.runPluginButton)
        self.plugin_dock.setWidget(pluginWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plugin_dock)
        self.plugin_dock.hide()

    def setupROIToolsDock(self):
        self.roi_tools_dock = QDockWidget("ROI Tools", self)
        roi_tools_widget = QWidget()
        roi_tools_layout = QVBoxLayout(roi_tools_widget)

        # ROI info widget
        self.roi_info_widget = ROIInfoWidget()
        roi_tools_layout.addWidget(self.roi_info_widget)

        # ROI Z-profile widget
        self.roi_z_profile_widget = ROIZProfileWidget()
        roi_tools_layout.addWidget(self.roi_z_profile_widget)

        # ROI zoom view
        self.roi_zoom_view = pg.ImageView()
        roi_tools_layout.addWidget(self.roi_zoom_view)

        self.roi_tools_dock.setWidget(roi_tools_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.roi_tools_dock)
        self.roi_tools_dock.hide()

    def setup_logging(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('microview.log', maxBytes=1e6, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.recent_files = config.get('recent_files', [])
                self.auto_load_plugins = config.get('auto_load_plugins', [])
        else:
            self.recent_files = []
            self.auto_load_plugins = []



    def createToolbar(self):
        self.toolbar = QToolBar()
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        self.toolbar.addAction('Undo', self.undo)
        self.toolbar.addAction('Redo', self.redo)
        self.toolbar.addSeparator()
        self.toolbar.addAction('Threshold', self.threshold)
        self.toolbar.addAction('Measure', self.measure)

        self.toggle_chart_button = QPushButton("Toggle Results Chart")
        self.toggle_chart_button.setCheckable(True)
        self.toggle_chart_button.toggled.connect(self.toggle_results_chart)
        self.toggle_chart_button.setEnabled(False)

        self.toggle_centroids_button = QPushButton("Toggle Centroids")
        self.toggle_centroids_button.setCheckable(True)
        self.toggle_centroids_button.toggled.connect(self.toggle_centroids)
        self.toggle_centroids_button.setEnabled(False)

        self.toolbar.addWidget(self.toggle_chart_button)
        self.toolbar.addWidget(self.toggle_centroids_button)


    def safe_disconnect(self, signal, slot):
        try:
            signal.disconnect(slot)
        except TypeError:
            # Signal was not connected
            pass

    def update_mouse_position(self, pos):
        if self.window_manager.current_window:
            image_pos = self.window_manager.current_window.imageView.getImageItem().mapFromScene(pos)
            # ... rest of the method
            x, y = int(image_pos.x()), int(image_pos.y())

            image = self.window_manager.current_window.image

            if image.ndim == 3:
                if 0 <= x < image.shape[2] and 0 <= y < image.shape[1]:
                    self.info_panel.update_mouse_info(x, y)
                    current_frame = self.window_manager.current_window.currentIndex
                    intensity = image[current_frame, y, x]
                    self.info_panel.update_intensity(intensity)
                    self.z_profile_widget.update_profile(image, x, y)
                else:
                    self.info_panel.update_mouse_info(None, None)
                    self.info_panel.update_intensity(None)
                    self.z_profile_widget.clear_profile()
            elif image.ndim == 2:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    self.info_panel.update_mouse_info(x, y)
                    intensity = image[y, x]
                    self.info_panel.update_intensity(intensity)
                    self.z_profile_widget.update_profile(image, x, y)
                else:
                    self.info_panel.update_mouse_info(None, None)
                    self.info_panel.update_intensity(None)
                    self.z_profile_widget.clear_profile()

    def update_roi_info(self, roi):
        if roi is not None and self.window_manager.current_window:
            roi_data = roi.get_roi_data()
            self.roi_info_widget.update_roi_info(roi_data)
            self.roi_zoom_view.setImage(roi_data)
            if isinstance(roi, LineROI):
                roi.update_profile()
        else:
            self.roi_info_widget.update_roi_info(None)
            self.roi_zoom_view.clear()


    def update_frame_info(self, frame):
        if self.window_manager.current_window:
            self.info_panel.update_info(self.window_manager.current_window)
            self.window_manager.current_window.update_status_bar()

    def get_plugins(self):
        return self.plugins


    def undo(self):
        # Implement undo functionality
        pass

    def redo(self):
        # Implement redo functionality
        pass

    def particleAnalysis(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            threshold = filters.threshold_otsu(image)
            binary = image > threshold
            labeled = measure.label(binary)
            props = measure.regionprops(labeled)
            for prop in props:
                print(f"Area: {prop.area}, Centroid: {prop.centroid}")

    def on_particle_analysis_complete(self, df):
        self.particle_analysis_results = df
        print(f"Received particle analysis results with {len(df)} particles")  # Debug print
        print(f"Columns in results: {df.columns}")  # Debug print
        self.toggle_chart_button.setEnabled(True)
        self.toggle_centroids_button.setEnabled(True)
        QMessageBox.information(self, "Analysis Complete", f"Found {len(df)} particles.")


    def show_results_chart(self):
        if not hasattr(self, 'results_chart_window'):
            self.results_chart_window = QWidget()
            layout = QVBoxLayout()

            # Create table view
            table_view = QTableView()
            model = PandasModel(self.particle_analysis_results)
            table_view.setModel(model)
            layout.addWidget(table_view)

            # Create scatter plot
            scatter_widget = pg.PlotWidget()
            scatter_widget.setLabel('left', 'Y')
            scatter_widget.setLabel('bottom', 'X')
            scatter_widget.setTitle('Particle Locations and Trajectories')

            # Plot particles
            particles = self.particle_analysis_results
            is_linked = 'particle' in particles.columns

            print(f"Total particles: {len(particles)}")  # Debug print
            print(f"Columns: {particles.columns.tolist()}")  # Debug print
            print(f"First few rows:\n{particles.head()}")  # Debug print

            max_particles_to_plot = 10000  # Limit the number of particles to plot for performance

            if is_linked:
                unique_particles = particles['particle'].unique()
                print(f"Unique particles: {len(unique_particles)}")  # Debug print

                # Sample particles if there are too many
                if len(unique_particles) > max_particles_to_plot:
                    unique_particles = np.random.choice(unique_particles, max_particles_to_plot, replace=False)

                scatter_plot = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
                for particle_id in unique_particles:
                    particle_data = particles[particles['particle'] == particle_id]
                    if not particle_data.empty:
                        color = pg.intColor(particle_id, hues=len(unique_particles))
                        x = particle_data['centroid-1'].values
                        y = particle_data['centroid-0'].values
                        scatter_plot.addPoints(x=x, y=y, brush=color)
                scatter_widget.addItem(scatter_plot)
            else:
                # Sample particles if there are too many
                if len(particles) > max_particles_to_plot:
                    particles = particles.sample(max_particles_to_plot)

                x = particles['centroid-1'].values
                y = particles['centroid-0'].values
                scatter_plot = pg.ScatterPlotItem(x=x, y=y, size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
                scatter_widget.addItem(scatter_plot)

            layout.addWidget(scatter_widget)

            self.results_chart_window.setLayout(layout)
            self.results_chart_window.setWindowTitle("Particle Analysis Results")
            self.results_chart_window.resize(800, 600)

        self.results_chart_window.show()

    def hide_results_chart(self):
        if hasattr(self, 'results_chart_window'):
            self.results_chart_window.hide()

    def remove_centroids(self, window):
        if hasattr(window, 'centroid_items'):
            for item in window.centroid_items:
                window.get_view().removeItem(item)
            window.centroid_items.clear()

    def plot_centroids(self, window):
        if not hasattr(window, 'centroid_items'):
            window.centroid_items = []

        self.remove_centroids(window)  # Clear existing centroids

        current_frame = window.get_current_frame()
        frame_particles = self.particle_analysis_results[self.particle_analysis_results['frame'] == current_frame]

        print(f"Plotting {len(frame_particles)} particles for frame {current_frame}")  # Debug print

        is_trackpy = 'particle' in self.particle_analysis_results.columns  # Check if trackpy was used

        for _, row in frame_particles.iterrows():
            color = pg.intColor(row['particle'], hues=50, alpha=120) if is_trackpy else pg.mkBrush(255, 0, 0, 120)

            centroid = pg.ScatterPlotItem([row['centroid-1']], [row['centroid-0']], size=10, pen=pg.mkPen(None), brush=color)
            window.get_view().addItem(centroid)
            window.centroid_items.append(centroid)

            if is_trackpy:
                # Add trajectory if trackpy was used
                trajectory = self.particle_analysis_results[self.particle_analysis_results['particle'] == row['particle']]
                trajectory = trajectory[trajectory['frame'] <= current_frame]  # Only show up to current frame
                if len(trajectory) > 1:
                    trajectory_item = pg.PlotDataItem(trajectory['centroid-1'], trajectory['centroid-0'], pen=color)
                    window.get_view().addItem(trajectory_item)
                    window.centroid_items.append(trajectory_item)

        if hasattr(self, 'particle_count_label'):
            particle_count = len(frame_particles)
            self.particle_count_label.setText(f"Particles in frame: {particle_count}")

    def removeROI(self, roi):
        if self.window_manager.current_window:
            try:
                image_window = self.window_manager.current_window
                image_view = image_window.imageView
                image_window.getView().removeItem(roi)
                image_window.rois.remove(roi)

                # Disconnect the ROI from the time slider if it exists
                if hasattr(image_view, 'timeLine') and isinstance(roi, LineROI):
                    image_view.timeLine.sigPositionChanged.disconnect(roi.update_profile)

                # Clear ROI info and zoom view
                self.update_roi_info(None)

                print("ROI removed")
            except Exception as e:
                print(f"Error removing ROI: {str(e)}")

    def loadPlugin(self):
        pluginName, _ = QFileDialog.getOpenFileName(self, "Load Plugin", "", "Python Files (*.py)")
        if pluginName:
            try:
                logger.info(f"Attempting to load plugin from file: {pluginName}")

                # Add the plugin's directory to sys.path
                plugin_dir = os.path.dirname(os.path.abspath(pluginName))
                if plugin_dir not in sys.path:
                    sys.path.insert(0, plugin_dir)

                # Use a unique name for each loaded module to avoid conflicts
                module_name = f"plugin_module_{os.path.basename(pluginName)}"
                spec = importlib.util.spec_from_file_location(module_name, pluginName)
                plugin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plugin_module)

                if hasattr(plugin_module, 'Plugin'):
                    plugin_class = plugin_module.Plugin
                    plugin = plugin_class(self)
                    self.plugins[plugin.name] = plugin
                    self.pluginList.addItem(plugin.name)  # This line stays the same
                    logger.info(f"Successfully loaded plugin: {plugin.name}")
                else:
                    raise AttributeError("Module does not contain a 'Plugin' class")

            except Exception as e:
                logger.error(f"Error loading plugin from {pluginName}: {str(e)}")
                logger.error(traceback.format_exc())
                print(f"Error loading plugin: {str(e)}")
            finally:
                # Remove the plugin's directory from sys.path to avoid potential conflicts
                if plugin_dir in sys.path:
                    sys.path.remove(plugin_dir)

    def togglePluginWindow(self):
        if self.plugin_dock.isVisible():
            self.plugin_dock.hide()
        else:
            self.plugin_dock.show()


    def flika_open_file(self, filename):
        return flika_open_file(self, filename)

    def flika_open_rois(self, filename):
        return flika_open_rois(self, filename)

    def open_rois(self, filename):
        try:
            with open(filename, 'r') as f:
                roi_data = json.load(f)

            rois = []
            current_window = self.window_manager.current_window

            if current_window is None:
                logger.warning("No current window to add ROIs to.")
                return rois

            for roi_info in roi_data:
                roi_type = roi_info['type']
                pos = roi_info['pos']
                size = roi_info['size']

                if roi_type == 'rectangle':
                    roi = RectROI(pos, size, current_window, pen='r')
                elif roi_type == 'ellipse':
                    roi = EllipseROI(pos, size, current_window, pen='r')
                elif roi_type == 'line':
                    roi = LineROI([pos, [pos[0] + size[0], pos[1] + size[1]]], current_window, pen='r')
                else:
                    logger.warning(f"Unknown ROI type: {roi_type}")
                    continue

                current_window.view.addItem(roi)
                roi.sigRegionChangeFinished.connect(lambda roi=roi: self.roiChanged(roi))
                roi.sigRemoveRequested.connect(lambda roi=roi: self.removeROI(roi))

                if not hasattr(current_window, 'rois'):
                    current_window.rois = []
                current_window.rois.append(roi)
                rois.append(roi)

            logger.info(f"Loaded {len(rois)} ROIs from {filename}")
            return rois
        except Exception as e:
            logger.error(f"Error loading ROIs from {filename}: {str(e)}")
            return []

    def save_rois(self, filename):
        try:
            current_window = self.window_manager.current_window
            if current_window is None or not hasattr(current_window, 'rois'):
                logger.warning("No ROIs to save.")
                return

            roi_data = []
            for roi in current_window.rois:
                roi_info = {
                    'type': roi.__class__.__name__.lower().replace('roi', ''),
                    'pos': roi.pos().tolist() if hasattr(roi, 'pos') else roi.getState()['pos'],
                    'size': roi.size().tolist() if hasattr(roi, 'size') else roi.getState()['size'],
                }
                roi_data.append(roi_info)

            with open(filename, 'w') as f:
                json.dump(roi_data, f)

            logger.info(f"Saved {len(roi_data)} ROIs to {filename}")
        except Exception as e:
            logger.error(f"Error saving ROIs to {filename}: {str(e)}")


    def add_flika_window(self, flika_window):
        self.flika_windows.append(flika_window)
        # Instead of adding the FlikaMicroViewWindow directly, we'll add its Flika window
        self.window_manager.add_window(flika_window.flika_window)
        # Connect Flika window signals to MicroView slots if needed
        flika_window.timeChanged.connect(self.on_time_slider_changed)


    def closeEvent(self, event):
        # This method is called when the window is about to be closed
        self.close_all_plugins()
        super().closeEvent(event)

    def createInfoDock(self):
        self.info_panel = InfoPanel()
        self.info_dock = QDockWidget("Image Information", self)
        self.info_dock.setWidget(self.info_panel)
        self.info_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.info_dock.setMinimumWidth(250)  # Set a minimum width for the info dock

    def adjustDockSizes(self):
        width = self.width()
        height = self.height()

        # Adjust top row (info, ROI info, histogram) to take up about 30% of the height
        top_height = int(height * 0.3)
        self.info_dock.setMaximumHeight(top_height)
        if hasattr(self, 'roi_info_widget'):
            self.roi_info_widget.setMaximumHeight(top_height)

        # Adjust z-profile widgets to take up about 20% of the height each
        z_profile_height = int(height * 0.2)
        if hasattr(self, 'z_profile_widget'):
            self.z_profile_widget.setMinimumHeight(z_profile_height)
        if hasattr(self, 'roi_z_profile_widget'):
            self.roi_z_profile_widget.setMinimumHeight(z_profile_height)

        # Set a maximum width for the info dock
        max_info_width = int(width * 0.3)  # 30% of the window width
        self.info_dock.setMaximumWidth(max_info_width)

    def createZProfileDock(self):
        self.z_profile_widget = ZProfileWidget()
        self.z_profile_dock = QDockWidget("Z-Profile", self)
        self.z_profile_dock.setWidget(self.z_profile_widget)
        self.z_profile_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

    def createPluginDock(self):
        plugin_widget = QWidget()
        plugin_layout = QVBoxLayout(plugin_widget)
        self.pluginList = QListWidget()
        self.pluginList.itemDoubleClicked.connect(self.runSelectedPlugin)
        plugin_layout.addWidget(self.pluginList)
        self.runPluginButton = QPushButton("Run Selected Plugin")
        self.runPluginButton.clicked.connect(self.runSelectedPlugin)
        plugin_layout.addWidget(self.runPluginButton)

        self.plugin_dock = QDockWidget("Plugins", self)
        self.plugin_dock.setWidget(plugin_widget)
        self.plugin_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

    def createROIToolsDock(self):
        roi_tools_widget = QWidget()
        roi_tools_layout = QVBoxLayout(roi_tools_widget)

        self.roi_info_widget = ROIInfoWidget()
        roi_tools_layout.addWidget(self.roi_info_widget)

        self.roi_z_profile_widget = ROIZProfileWidget()
        roi_tools_layout.addWidget(self.roi_z_profile_widget)

        self.roi_zoom_view = pg.ImageView()
        roi_tools_layout.addWidget(self.roi_zoom_view)

        self.roi_tools_dock = QDockWidget("ROI Tools", self)
        self.roi_tools_dock.setWidget(roi_tools_widget)
        self.roi_tools_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

    def createConsoleDock(self):
        if not self.in_spyder:
            self.console = MicroViewConsole(self)
            self.console_dock = QDockWidget("IPython Console", self)
            self.console_dock.setWidget(self.console)
            self.console_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        else:
            self.console = None
            self.console_dock = None

    def toggleROIToolsDock(self):
        if self.roi_tools_dock.isVisible():
            self.roi_tools_dock.hide()
        else:
            self.roi_tools_dock.show()

    def togglePluginDock(self):
        if self.plugin_dock.isVisible():
            self.plugin_dock.hide()
        else:
            self.plugin_dock.show()

    def setupMenus(self):
        self.menu_manager.create_menus()

    def display_analysis_result(self, result):
        print("Displaying result in MicroView")
        print(f"Result in MicroView - Shape: {result.shape}, dtype: {result.dtype}")
        print(f"Result stats - Min: {np.min(result)}, Max: {np.max(result)}, Mean: {np.mean(result)}")
        window = ImageWindow(result, "Analysis Result")
        self.window_manager.add_window(window)
        self.set_current_window(window)

    def open_transformations_dialog(self):
        if self.window_manager.current_window:
            dialog = TransformationsDialog(self.window_manager.current_window.image, self)
            dialog.transformationApplied.connect(self.apply_transformation)
            dialog.exec_()

    def apply_transformation(self, transformed_data):
        if self.window_manager.current_window:
            self.window_manager.current_window.setImage(transformed_data)
            self.info_panel.update_info(self.window_manager.current_window)

    def open_synthetic_data_dialog(self):
        dialog = SyntheticDataDialog(self)
        dialog.dataGenerated.connect(self.create_synthetic_data_window)
        dialog.exec_()

    def create_synthetic_data_window(self, data):
        window = ImageWindow(data, "Synthetic Data")
        self.window_manager.add_window(window)
        self.set_current_window(window)



# At the end of the file, after the MicroView class definition
def initialize_microview():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    ex = MicroView()
    return ex

# Main execution
if __name__ == '__main__':
    logger.info("Starting MicroView application")
    from global_vars import g
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Check if we're running in Spyder
    in_spyder = get_ipython().__class__.__name__ == 'SpyderShell'

    # Create QApplication instance
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create and show MicroView instance
    ex = initialize_microview()
    g.set_microview(ex)
    ex.show()

    # If in Spyder, provide a helper function to access shared variables
    if ex.in_spyder:
        def get_mv_var(name):
            return ex.get_variable(name)
        print("MicroView is ready. Use get_mv_var(name) to access shared variables.")
        print("Type ex.show_ipython_commands() for available commands.")

    # Run the event loop if not in Spyder
    if not in_spyder:
        sys.exit(app.exec_())
    else:
        # If in Spyder, we don't need to start the event loop
        print("MicroView is ready. You can interact with 'ex' object in the Spyder console.")
