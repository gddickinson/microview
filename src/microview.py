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
    current_window_changed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.plugins = {}
        self.recent_files = []
        self.config_file = os.path.expanduser('~/.microview_config.json')
        self.setup_logging()
        self.load_config()
        self.in_spyder = get_ipython().__class__.__name__ == 'SpyderShell'
        self.filters = Filters(self)
        self.particle_analysis_results = None
        self.menu_manager = MenuManager(self)
        self.image_processor = ImageProcessor()
        self.window_manager = WindowManager()
        self.file_loader = FileLoader()
        self.shared_variables = {}
        self.initUI()
        self.load_plugins()
        logger.info(f"Plugins loaded: {list(self.plugins.keys())}")
        self.flika_windows = []
        #self.current_window = None
        self.windows = []
        self.setupMenus()

    def initUI(self):
        self.setWindowTitle('MicroView Control Panel')
        self.setGeometry(100, 100, 1200, 800)

        self.createToolbar()

        # Create dock widgets
        self.createInfoDock()
        self.createZProfileDock()
        self.createPluginDock()
        self.createROIToolsDock()
        self.createConsoleDock()

        # Arrange docks
        self.addDockWidget(Qt.TopDockWidgetArea, self.info_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.z_profile_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.roi_tools_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plugin_dock)
        if self.console_dock:
            self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)

        # Hide some docks by default
        self.roi_tools_dock.hide()
        self.plugin_dock.hide()

        # Particle count label in status bar
        self.particle_count_label = QLabel("Particles in frame: 0")
        self.statusBar().addPermanentWidget(self.particle_count_label)

        # Connect signals
        self.current_window_changed.connect(self.info_panel.update_info)

        # Initialize shared variables
        self.push_variables({
            'np': np,
            'pg': pg,
            'filters': filters,
            'morphology': morphology,
            'measure': measure,
            'segmentation': segmentation,
            'g': g
        })

        self.show()

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

    def save_config(self):
        config = {
            'recent_files': self.recent_files,
            'auto_load_plugins': self.auto_load_plugins
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f)

    def update_recent_files_menu(self):
        self.menu_manager.update_recent_files_menu()

    def push_variables(self, variables):
        self.shared_variables.update(variables)
        if self.console is not None:
            self.console.push_variables(variables)
        logger.info(f"Updated shared variables: {', '.join(variables.keys())}")

    def get_variable(self, name):
        return self.shared_variables.get(name)

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

    def set_current_window(self, window):
        if self.window_manager.current_window:
            # Safely disconnect signals from the old current window
            self.safe_disconnect(self.window_manager.current_window.imageView.scene.sigMouseMoved, self.update_mouse_position)
            self.safe_disconnect(self.window_manager.current_window.timeChanged, self.update_frame_info)
            self.safe_disconnect(self.window_manager.current_window.roiChanged, self.update_roi_info)
            self.window_manager.current_window.set_as_current(False)

        self.window_manager.current_window = window
        self.current_window_changed.emit(window)

        if window:
            # Connect signals to the new current window
            window.imageView.scene.sigMouseMoved.connect(self.update_mouse_position)
            window.timeChanged.connect(self.update_frame_info)
            window.roiChanged.connect(self.update_roi_info)
            window.set_as_current(True)

        self.update_frame_info(0)
        self.update_roi_info(None)  # Clear ROI info when changing windows


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


    def openFile(self):
        file_types = "All Supported Files ("
        file_types += " ".join(f"*{ext}" for ext in self.file_loader.supported_extensions)
        file_types += ");;"
        file_types += ";;".join([f"{ext.upper()[1:]} Files (*{ext})" for ext in self.file_loader.supported_extensions])

        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", file_types)
        if fileName:
            self.loadImage(fileName)


    def loadImage(self, fileName):
        try:
            result = self.file_loader.load_file(fileName)
            if result is not None:
                image, metadata = result

                # Ensure metadata is a dictionary
                metadata = metadata or {}

                # Save metadata to a separate file
                try:
                    self.file_loader.save_metadata(fileName, metadata)
                except Exception as e:
                    logger.warning(f"Failed to save metadata: {str(e)}")

                window = ImageWindow(image, os.path.basename(fileName), metadata)
                self.window_manager.add_window(window)
                self.set_current_window(window)
                self.windows.append(window)
                window.windowSelected.connect(self.set_current_window)

                # Connect the time changed signal
                window.timeChanged.connect(self.on_time_slider_changed)

                # Update shared variables
                self.push_variables({
                    'current_image': image,
                    'current_metadata': metadata,
                    'current_window': window
                })

                logger.info(f"Loaded image: {fileName}")

                # Update recent files
                if fileName in self.recent_files:
                    self.recent_files.remove(fileName)
                self.recent_files.insert(0, fileName)
                self.recent_files = self.recent_files[:10]  # Keep only 10 most recent
                self.update_recent_files_menu()
                self.save_config()

        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")


    def display_metadata_info(self, metadata):
        info = f"Image dimensions: {metadata['dims']}\n"
        info += f"Image shape: {metadata['shape']}\n"
        info += f"Data type: {metadata['dtype']}\n"

        if metadata['pixel_size_um']:
            info += f"Pixel size: {metadata['pixel_size_um']} µm\n"
        if metadata['time_interval_s']:
            info += f"Time interval: {metadata['time_interval_s']} s\n"
        if metadata['channel_names']:
            info += f"Channels: {', '.join(metadata['channel_names'])}\n"

        QMessageBox.information(self, "Image Metadata", info)


    def load_plugins(self):
        logger.info("Starting to load plugins")
        plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
        logger.info(f"Plugins directory: {plugins_dir}")
        startup_file = os.path.join(plugins_dir, 'startup.json')
        logger.info(f"Startup file path: {startup_file}")

        if os.path.exists(startup_file):
            with open(startup_file, 'r') as f:
                startup_config = json.load(f)
            enabled_plugins = startup_config.get('enabled_plugins', [])
            logger.info(f"Enabled plugins from startup file: {enabled_plugins}")
        else:
            enabled_plugins = []
            logger.warning("Startup file not found. All plugins will be loaded.")

        for item in os.listdir(plugins_dir):
            plugin_dir = os.path.join(plugins_dir, item)
            logger.info(f"Checking directory: {plugin_dir}")
            if os.path.isdir(plugin_dir):
                plugin_file = os.path.join(plugin_dir, f"{item}.py")
                logger.info(f"Looking for plugin file: {plugin_file}")
                if os.path.exists(plugin_file):
                    plugin_name = item
                    logger.info(f"Found plugin: {plugin_name}")
                    if plugin_name in enabled_plugins or not enabled_plugins:
                        try:
                            logger.info(f"Attempting to load plugin: {plugin_name}")
                            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            plugin_class = getattr(module, 'Plugin')
                            plugin = plugin_class(self)
                            self.plugins[plugin.name] = plugin
                            logger.info(f"Successfully loaded plugin: {plugin.name}")
                        except Exception as e:
                            logger.error(f"Error loading plugin {plugin_name}: {str(e)}")
                            logger.error(traceback.format_exc())
                    else:
                        logger.info(f"Plugin {plugin_name} is not enabled in startup.json")
                else:
                    logger.warning(f"Plugin file not found: {plugin_file}")
            else:
                logger.info(f"Not a directory: {plugin_dir}")

        logger.info(f"Finished loading plugins. Total plugins loaded: {len(self.plugins)}")
        self.update_plugin_list()

    def update_plugin_list(self):
        self.pluginList.clear()
        for plugin_name in self.plugins.keys():
            self.pluginList.addItem(plugin_name)

    def get_plugins(self):
        return self.plugins

    def saveFile(self):
        if self.window_manager.current_window:
            fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "TIFF Files (*.tiff)")
            if fileName:
                tifffile.imwrite(fileName, self.window_manager.current_window.image)

    def closeCurrentWindow(self):
        self.window_manager.close_current_window()

    def undo(self):
        # Implement undo functionality
        pass

    def redo(self):
        # Implement redo functionality
        pass



    def mathOperation(self, operation):
        if self.window_manager.current_window:
            value, ok = QInputDialog.getDouble(self, "Input", "Enter value:")
            if ok:
                image = self.window_manager.current_window.image
                original_dtype = image.dtype

                # Convert image to float64 for calculations
                image = image.astype(np.float64)

                if operation == 'add':
                    result = image + value
                elif operation == 'subtract':
                    result = image - value
                elif operation == 'multiply':
                    result = image * value
                elif operation == 'divide':
                    # Avoid division by zero
                    if value == 0:
                        QMessageBox.warning(self, "Error", "Cannot divide by zero.")
                        return
                    result = image / value
                else:
                    QMessageBox.warning(self, "Error", f"Unknown operation: {operation}")
                    return

                # Clip the result to the range of the original dtype
                info = np.iinfo(original_dtype)
                result = np.clip(result, info.min, info.max)

                # Convert back to the original dtype
                result = result.astype(original_dtype)

                self.window_manager.current_window.setImage(result)
                print(f"Applied {operation} operation with value {value}")
        else:
            QMessageBox.warning(self, "Error", "No image window is currently active.")

    def gaussianBlur(self):
        if self.window_manager.current_window:
            sigma, ok = QInputDialog.getDouble(self, "Gaussian Blur", "Enter sigma value:")
            if ok:
                image = self.window_manager.current_window.image
                blurred = self.image_processor.gaussian_blur(image, sigma)
                self.window_manager.current_window.setImage(blurred)

    def medianFilter(self):
        if self.window_manager.current_window:
            size, ok = QInputDialog.getInt(self, "Median Filter", "Enter filter size:")
            if ok:
                image = self.window_manager.current_window.image
                filtered = self.image_processor.median_filter(image, size)
                self.window_manager.current_window.setImage(filtered)

    def sobelEdge(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            edges = self.image_processor.sobel_edge(image)
            self.window_manager.current_window.setImage(edges)

    def threshold(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            binary = self.image_processor.threshold(image)
            self.window_manager.current_window.setImage(binary)

    def erode(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            eroded = self.image_processor.erode(image)
            self.window_manager.current_window.setImage(eroded)

    def dilate(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            dilated = self.image_processor.dilate(image)
            self.window_manager.current_window.setImage(dilated)

    def zProjectMax(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            if image.ndim == 3:
                try:
                    projected = self.image_processor.z_project(image, method='max')
                    self.window_manager.current_window.setImage(projected)
                except Exception as e:
                    self.logger.error(f"Error in zProjectMax: {str(e)}")
                    QMessageBox.critical(self, "Error", f"Error in maximum intensity projection: {str(e)}")

    def zProjectMean(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            if image.ndim == 3:
                try:
                    projected = self.image_processor.z_project(image, method='mean')
                    self.window_manager.current_window.setImage(projected)
                except Exception as e:
                    self.logger.error(f"Error in zProjectMean: {str(e)}")
                    QMessageBox.critical(self, "Error", f"Error in mean intensity projection: {str(e)}")

    def measure(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            print(f"Mean: {np.mean(image)}")
            print(f"Std Dev: {np.std(image)}")
            print(f"Min: {np.min(image)}")
            print(f"Max: {np.max(image)}")

    def findMaxima(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            local_max = filters.peak_local_max(image)
            print(f"Found {len(local_max)} local maxima")

    def particleAnalysis(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            threshold = filters.threshold_otsu(image)
            binary = image > threshold
            labeled = measure.label(binary)
            props = measure.regionprops(labeled)
            for prop in props:
                print(f"Area: {prop.area}, Centroid: {prop.centroid}")


    def run_particle_analysis(self):
        if self.window_manager.current_window is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        try:
            image = self.window_manager.current_window.image
            analysis_dialog = ParticleAnalysisResults(self, image)
            analysis_dialog.analysisComplete.connect(self.on_particle_analysis_complete)
            analysis_dialog.exec_()  # Use exec_ instead of show() to make it modal

        except Exception as e:
            print(f"Error in particle analysis: {str(e)}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Error in particle analysis: {str(e)}")

    def on_particle_analysis_complete(self, df):
        self.particle_analysis_results = df
        print(f"Received particle analysis results with {len(df)} particles")  # Debug print
        print(f"Columns in results: {df.columns}")  # Debug print
        self.toggle_chart_button.setEnabled(True)
        self.toggle_centroids_button.setEnabled(True)
        QMessageBox.information(self, "Analysis Complete", f"Found {len(df)} particles.")


    def toggle_results_chart(self, checked):
        if self.particle_analysis_results is not None:
            if checked:
                self.show_results_chart()
            else:
                self.hide_results_chart()


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

    def toggle_centroids(self, checked):
        if self.particle_analysis_results is not None:
            current_window = self.window_manager.current_window
            if current_window:
                if checked:
                    self.plot_centroids(current_window)
                else:
                    self.remove_centroids(current_window)



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

    def on_time_slider_changed(self):
        if self.toggle_centroids_button.isChecked():
            current_window = self.window_manager.current_window
            if current_window:
                self.plot_centroids(current_window)


    def addROI(self, roi):
        if self.window_manager.current_window:
            try:
                image_window = self.window_manager.current_window

                image_window.add_roi(roi)
                print(f"ROI added to view: {roi}")  # Debug print
            except Exception as e:
                print(f"Error adding ROI: {str(e)}")

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

    def removeAllROIs(self):
        if self.window_manager.current_window:
            try:
                image_window = self.window_manager.current_window
                for roi in image_window.rois[:]:
                    self.removeROI(roi)
                print("All ROIs removed")  # Debug print
            except Exception as e:
                print(f"Error removing all ROIs: {str(e)}")

    def colocalization_analysis(self):
        if self.window_manager.current_window and hasattr(self.window_manager.current_window, 'rois') and len(self.window_manager.current_window.rois) == 1:
            roi = self.window_manager.current_window.rois[0]
            image_view = self.window_manager.current_window
            image = image_view.getImageItem().image

            if image.ndim != 3 or image.shape[2] != 2:
                print("Colocalization analysis requires a two-channel image")
                return

            roi_data = roi.getArrayRegion(image, image_view.getImageItem())
            channel1 = roi_data[:, :, 0].flatten()
            channel2 = roi_data[:, :, 1].flatten()

            pearson_corr, _ = stats.pearsonr(channel1, channel2)
            manders_m1 = np.sum(channel1[channel2 > 0]) / np.sum(channel1)
            manders_m2 = np.sum(channel2[channel1 > 0]) / np.sum(channel2)

            print("Colocalization Analysis:")
            print(f"Pearson's correlation coefficient: {pearson_corr:.3f}")
            print(f"Manders' coefficient M1: {manders_m1:.3f}")
            print(f"Manders' coefficient M2: {manders_m2:.3f}")

            # Create a scatter plot of the two channels
            scatter_plot = pg.plot(title="Channel Intensity Scatter Plot")
            scatter_plot.plot(channel1, channel2, pen=None, symbol='o', symbolSize=5, symbolPen=None, symbolBrush=(100, 100, 255, 50))
            scatter_plot.setLabel('left', 'Channel 2 Intensity')
            scatter_plot.setLabel('bottom', 'Channel 1 Intensity')
        else:
            print("Please select a single ROI for colocalization analysis")




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

    def runSelectedPlugin(self):
        selected_items = self.pluginList.selectedItems()
        if selected_items:
            plugin_name = selected_items[0].text()
            if plugin_name in self.plugins:
                self.plugins[plugin_name].run()
            else:
                print(f"Plugin {plugin_name} not found.")
        else:
            print("No plugin selected.")

    def togglePluginWindow(self):
        if self.plugin_dock.isVisible():
            self.plugin_dock.hide()
        else:
            self.plugin_dock.show()

    def tileWindows(self):
        self.window_manager.tile_windows()

    def cascadeWindows(self):
        self.window_manager.cascade_windows()

    def toggleConsole(self):
        if hasattr(self, 'console_dock') and self.console_dock.isVisible():
            self.console_dock.hide()
        elif hasattr(self, 'console_dock'):
            self.console_dock.show()

    def show_user_guide(self):
        QMessageBox.information(self, "User Guide", "User guide content goes here.")

    def show_ipython_commands(self):
        commands = """
        Available IPython commands:
        - get_mv_var(name): Get a shared variable
        - ex: The MicroView instance
        - ex.loadImage(filename): Load an image file
        - ex.current_window: Access the current image window
        """
        QMessageBox.information(self, "IPython Commands", commands)

    def show_about(self):
        about_text = f"""
        MicroView version {self.VERSION}

        A microscopy image viewer and analysis tool.

        Developed by [Your Name/Organization]
        """
        QMessageBox.about(self, "About MicroView", about_text)

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

    def save_rois_dialog(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save ROIs", "", "JSON Files (*.json)")
        if filename:
            self.save_rois(filename)

    def load_rois_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load ROIs", "", "JSON Files (*.json)")
        if filename:
            self.open_rois(filename)

    def add_menu_item(self, menu_name, item_name, callback):
        if menu_name not in self.menu_manager.menus:
            self.menu_manager.menus[menu_name] = self.menuBar().addMenu(menu_name)
        action = QAction(item_name, self)
        action.triggered.connect(callback)
        self.menu_manager.menus[menu_name].addAction(action)

    def add_flika_window(self, flika_window):
        self.flika_windows.append(flika_window)
        # Instead of adding the FlikaMicroViewWindow directly, we'll add its Flika window
        self.window_manager.add_window(flika_window.flika_window)
        # Connect Flika window signals to MicroView slots if needed
        flika_window.timeChanged.connect(self.on_time_slider_changed)

    def add_window(self, image, title=""):
        if isinstance(image, FlikaMicroViewWindow):
            window = image.flika_window
        else:
            window = ImageWindow(image, title)
        self.window_manager.add_window(window)

        # Connect the time changed signal
        if isinstance(image, FlikaMicroViewWindow):
            image.timeChanged.connect(self.on_time_slider_changed)
        elif hasattr(window, 'timeChanged'):
            window.timeChanged.connect(self.on_time_slider_changed)

        # Update shared variables
        self.push_variables({
            'current_image': window.image if hasattr(window, 'image') else image,
            'current_window': window
        })

        logger.info(f"Added new window: {title}")
        return window

    def closeEvent(self, event):
        # This method is called when the window is about to be closed
        self.close_all_plugins()
        super().closeEvent(event)

    def close_all_plugins(self):
        for plugin_name, plugin in self.plugins.items():
            try:
                if hasattr(plugin, 'close') and callable(plugin.close):
                    plugin.close()
                logger.info(f"Closed plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Error closing plugin {plugin_name}: {str(e)}")

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjustDockSizes()

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

    def open_analysis_console(self):
        if self.window_manager.current_window:
            image = self.window_manager.current_window.image
            self.analysis_console = ScikitAnalysisConsole(image)
            self.analysis_console.analysisCompleted.connect(self.display_analysis_result)
            self.analysis_console.show()

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
