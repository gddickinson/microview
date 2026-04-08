import sys
import numpy as np
import logging
import tifffile
import os
from typing import Tuple, List, Optional, Any, Dict

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QFileDialog, QMessageBox, QCheckBox, QDockWidget, QSizePolicy, QTableWidget,
                             QTableWidgetItem, QDialog, QGridLayout, QTabWidget, QTextEdit, QAction, QFormLayout,
                             QListWidget, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt, QSize
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer, QEvent
from PyQt5.QtGui import QColor, QVector3D, QImage, QMouseEvent, QWheelEvent
import traceback

from matplotlib import pyplot as plt
from skimage.feature import blob_log
from skimage import measure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from scipy.ndimage import distance_transform_edt, center_of_mass, binary_dilation, gaussian_filter
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import pyqtSlot

from data_generation import DataGenerator, DataManager
from blob_detection import BlobAnalyzer
from biological_simulation import BiologicalSimulator
from file_operations import ImporterFactory
#from volume_processor import VolumeProcessor
from biological_simulation import EnhancedBiologicalSimulator, Cell, CellularEnvironment, ShapeSimulator, Sphere, Cube, RandomWalk, LinearMotion, Attraction, Repulsion
from data_manager import DataManager
from visualization_manager import VisualizationManager
from ui_manager import UIManager
from raw_data_viewer import RawDataViewer
from data_properties_dialog import DataPropertiesDialog
from image_processing import ImageProcessor

import OpenGL
print(f"PyQtGraph version: {pg.__version__}")
print(f"OpenGL version: {OpenGL.__version__}")



class EnhancedBiologicalSimulationWidget(QWidget):
    simulationRequested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Create a widget to hold all the controls
        content_widget = QWidget()
        scroll.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)

        # Environment settings
        env_group = QGroupBox("Environment Settings")
        env_layout = QFormLayout()
        self.env_size_x = QSpinBox()
        self.env_size_y = QSpinBox()
        self.env_size_z = QSpinBox()
        for spinbox in [self.env_size_x, self.env_size_y, self.env_size_z]:
            spinbox.setRange(10, 500)
            spinbox.setValue(100)
        env_layout.addRow("Environment Size X:", self.env_size_x)
        env_layout.addRow("Environment Size Y:", self.env_size_y)
        env_layout.addRow("Environment Size Z:", self.env_size_z)
        env_group.setLayout(env_layout)
        content_layout.addWidget(env_group)

        # Cell settings
        cell_group = QGroupBox("Cell Settings")
        cell_layout = QFormLayout()
        self.num_cells = QSpinBox()
        self.num_cells.setRange(1, 20)
        self.num_cells.setValue(1)
        self.cell_type = QComboBox()
        self.cell_type.addItems(['spherical', 'neuron', 'epithelial', 'muscle'])
        cell_layout.addRow("Number of Cells:", self.num_cells)
        cell_layout.addRow("Cell Type:", self.cell_type)
        cell_group.setLayout(cell_layout)
        content_layout.addWidget(cell_group)

        # Protein dynamics
        protein_group = QGroupBox("Protein Dynamics")
        protein_layout = QFormLayout()
        self.diffusion_checkbox = QCheckBox()
        self.diffusion_coefficient = QDoubleSpinBox()
        self.diffusion_coefficient.setRange(0, 100)
        self.diffusion_coefficient.setValue(1)
        protein_layout.addRow("Enable Diffusion:", self.diffusion_checkbox)
        protein_layout.addRow("Diffusion Coefficient:", self.diffusion_coefficient)
        protein_group.setLayout(protein_layout)
        content_layout.addWidget(protein_group)

        # Calcium signaling
        calcium_group = QGroupBox("Calcium Signaling")
        calcium_layout = QFormLayout()
        self.calcium_signal_type = QComboBox()
        self.calcium_signal_type.addItems(['None', 'Wave', 'Spark'])
        self.calcium_amplitude = QDoubleSpinBox()
        self.calcium_amplitude.setRange(0, 10)
        self.calcium_amplitude.setValue(1)
        calcium_layout.addRow("Signal Type:", self.calcium_signal_type)
        calcium_layout.addRow("Amplitude:", self.calcium_amplitude)
        calcium_group.setLayout(calcium_layout)
        content_layout.addWidget(calcium_group)

        # Simulation button
        self.simulate_button = QPushButton("Run Simulation")
        self.simulate_button.clicked.connect(self.requestSimulation)
        layout.addWidget(self.simulate_button)

    def requestSimulation(self):
        params = {
            'environment_size': (self.env_size_x.value(), self.env_size_y.value(), self.env_size_z.value()),
            'num_cells': self.num_cells.value(),
            'cell_type': self.cell_type.currentText(),
            'protein_diffusion': {
                'enabled': self.diffusion_checkbox.isChecked(),
                'coefficient': self.diffusion_coefficient.value()
            },
            'calcium_signal': {
                'type': self.calcium_signal_type.currentText(),
                'amplitude': self.calcium_amplitude.value()
            }
        }
        self.simulationRequested.emit(params)

class BiologicalSimulationWidget(QWidget):
    simulationRequested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # Create a widget to hold all the content
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        content_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_protein_tab()
        self.create_structure_tab()
        self.create_organelles_tab()
        self.create_calcium_tab()

        # Simulate Button
        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.requestSimulation)
        main_layout.addWidget(self.simulate_button)

        # Set the size policy to allow the widget to be resized
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Set a minimum size for the widget
        self.setMinimumSize(400, 600)

    def create_protein_tab(self):
        protein_tab = QWidget()
        layout = QVBoxLayout(protein_tab)

        # Protein Dynamics group
        protein_group = QGroupBox("Protein Dynamics")
        protein_layout = QFormLayout(protein_group)

        self.diffusion_checkbox = QCheckBox()
        self.diffusion_coefficient = QDoubleSpinBox()
        self.diffusion_coefficient.setRange(0, 100)
        self.diffusion_coefficient.setValue(1)

        self.transport_checkbox = QCheckBox()
        self.transport_velocity = QDoubleSpinBox()
        self.transport_velocity.setRange(-10, 10)
        self.transport_velocity.setValue(1)

        protein_layout.addRow("Protein Diffusion:", self.diffusion_checkbox)
        protein_layout.addRow("Diffusion Coefficient:", self.diffusion_coefficient)
        protein_layout.addRow("Active Transport:", self.transport_checkbox)
        protein_layout.addRow("Transport Velocity:", self.transport_velocity)

        layout.addWidget(protein_group)
        self.tab_widget.addTab(protein_tab, "Protein Dynamics")

    def create_structure_tab(self):
        structure_tab = QWidget()
        layout = QVBoxLayout(structure_tab)

        # Cell Structure group
        structure_group = QGroupBox("Cell Structure")
        structure_layout = QFormLayout(structure_group)

        self.cell_type_combo = QComboBox()
        self.cell_type_combo.addItems(['spherical', 'neuron', 'epithelial', 'muscle'])
        self.cell_type_combo.currentTextChanged.connect(self.toggle_neuron_options)

        self.cell_membrane_checkbox = QCheckBox("Cell Membrane")
        self.nucleus_checkbox = QCheckBox("Nucleus")
        self.er_checkbox = QCheckBox("Endoplasmic Reticulum")
        self.mitochondria_checkbox = QCheckBox("Mitochondria")
        self.cytoskeleton_checkbox = QCheckBox("Cytoskeleton")

        structure_layout.addRow("Cell Type:", self.cell_type_combo)
        structure_layout.addRow(self.cell_membrane_checkbox)
        structure_layout.addRow(self.nucleus_checkbox)
        structure_layout.addRow(self.er_checkbox)
        structure_layout.addRow(self.mitochondria_checkbox)
        structure_layout.addRow(self.cytoskeleton_checkbox)

        layout.addWidget(structure_group)

        # Cell Parameters group
        params_group = QGroupBox("Cell Parameters")
        params_layout = QFormLayout(params_group)

        self.cell_radius = QSpinBox()
        self.cell_radius.setRange(5, 50)
        self.cell_radius.setValue(20)

        self.membrane_thickness = QSpinBox()
        self.membrane_thickness.setRange(1, 5)
        self.membrane_thickness.setValue(1)

        self.nucleus_radius = QSpinBox()
        self.nucleus_radius.setRange(1, 20)
        self.nucleus_radius.setValue(5)

        self.pixel_size_x = QDoubleSpinBox()
        self.pixel_size_y = QDoubleSpinBox()
        self.pixel_size_z = QDoubleSpinBox()
        for spinbox in [self.pixel_size_x, self.pixel_size_y, self.pixel_size_z]:
            spinbox.setRange(0.1, 10)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(1)

        self.er_density = QDoubleSpinBox()
        self.er_density.setRange(0.05, 0.2)
        self.er_density.setSingleStep(0.01)
        self.er_density.setValue(0.1)


        params_layout.addRow("Cell Radius:", self.cell_radius)
        params_layout.addRow("Membrane Thickness:", self.membrane_thickness)
        params_layout.addRow("Nucleus Radius:", self.nucleus_radius)
        params_layout.addRow("ER Density:", self.er_density)
        params_layout.addRow("Pixel Size X:", self.pixel_size_x)
        params_layout.addRow("Pixel Size Y:", self.pixel_size_y)
        params_layout.addRow("Pixel Size Z:", self.pixel_size_z)

        layout.addWidget(params_group)

        # Neuron-specific options
        self.neuron_options = QGroupBox("Neuron Options")
        neuron_layout = QFormLayout(self.neuron_options)

        self.soma_radius = QSpinBox()
        self.soma_radius.setRange(1, 20)
        self.soma_radius.setValue(5)

        self.axon_length = QSpinBox()
        self.axon_length.setRange(10, 100)
        self.axon_length.setValue(50)

        self.axon_width = QSpinBox()
        self.axon_width.setRange(1, 10)
        self.axon_width.setValue(2)

        self.num_dendrites = QSpinBox()
        self.num_dendrites.setRange(1, 10)
        self.num_dendrites.setValue(5)

        self.dendrite_length = QSpinBox()
        self.dendrite_length.setRange(5, 50)
        self.dendrite_length.setValue(25)

        neuron_layout.addRow("Soma Radius:", self.soma_radius)
        neuron_layout.addRow("Axon Length:", self.axon_length)
        neuron_layout.addRow("Axon Width:", self.axon_width)
        neuron_layout.addRow("Number of Dendrites:", self.num_dendrites)
        neuron_layout.addRow("Dendrite Length:", self.dendrite_length)

        layout.addWidget(self.neuron_options)
        self.neuron_options.setVisible(False)

        self.tab_widget.addTab(structure_tab, "Cell Structure")

    def create_organelles_tab(self):
        organelles_tab = QWidget()
        layout = QVBoxLayout(organelles_tab)

        # Mitochondria group
        mito_group = QGroupBox("Mitochondria")
        mito_layout = QFormLayout(mito_group)

        self.mito_count = QSpinBox()
        self.mito_count.setRange(10, 200)
        self.mito_count.setValue(50)

        self.mito_size_min = QSpinBox()
        self.mito_size_max = QSpinBox()
        self.mito_size_min.setRange(1, 10)
        self.mito_size_max.setRange(1, 10)
        self.mito_size_min.setValue(3)
        self.mito_size_max.setValue(8)

        mito_layout.addRow("Count:", self.mito_count)
        mito_layout.addRow("Min Size:", self.mito_size_min)
        mito_layout.addRow("Max Size:", self.mito_size_max)

        layout.addWidget(mito_group)

        # Cytoskeleton group
        cyto_group = QGroupBox("Cytoskeleton")
        cyto_layout = QFormLayout(cyto_group)

        self.actin_density = QDoubleSpinBox()
        self.actin_density.setRange(0.01, 0.1)
        self.actin_density.setSingleStep(0.01)
        self.actin_density.setValue(0.05)

        self.microtubule_density = QDoubleSpinBox()
        self.microtubule_density.setRange(0.01, 0.1)
        self.microtubule_density.setSingleStep(0.01)
        self.microtubule_density.setValue(0.02)

        cyto_layout.addRow("Actin Density:", self.actin_density)
        cyto_layout.addRow("Microtubule Density:", self.microtubule_density)

        layout.addWidget(cyto_group)

        self.tab_widget.addTab(organelles_tab, "Organelles")

    def create_calcium_tab(self):
        calcium_tab = QWidget()
        layout = QVBoxLayout(calcium_tab)

        # Calcium Signaling group
        calcium_group = QGroupBox("Calcium Signaling")
        calcium_layout = QFormLayout(calcium_group)

        self.calcium_combo = QComboBox()
        self.calcium_combo.addItems(['None', 'Blip', 'Puff', 'Wave'])

        self.calcium_intensity = QDoubleSpinBox()
        self.calcium_intensity.setRange(0, 1)
        self.calcium_intensity.setSingleStep(0.1)
        self.calcium_intensity.setValue(0.5)

        self.calcium_duration = QSpinBox()
        self.calcium_duration.setRange(1, 100)
        self.calcium_duration.setValue(10)

        calcium_layout.addRow("Signal Type:", self.calcium_combo)
        calcium_layout.addRow("Signal Intensity:", self.calcium_intensity)
        calcium_layout.addRow("Signal Duration:", self.calcium_duration)

        layout.addWidget(calcium_group)

        self.tab_widget.addTab(calcium_tab, "Calcium Signaling")

    def toggle_neuron_options(self, cell_type):
        self.neuron_options.setVisible(cell_type == 'neuron')

    def requestSimulation(self):
        params = {
            'protein_diffusion': {
                'enabled': self.diffusion_checkbox.isChecked(),
                'coefficient': self.diffusion_coefficient.value()
            },
            'active_transport': {
                'enabled': self.transport_checkbox.isChecked(),
                'velocity': self.transport_velocity.value()
            },
            'cellular_structures': {
                'cell_membrane': self.cell_membrane_checkbox.isChecked(),
                'nucleus': self.nucleus_checkbox.isChecked(),
                'er': self.er_checkbox.isChecked(),
                'mitochondria': self.mitochondria_checkbox.isChecked(),
                'cytoskeleton': self.cytoskeleton_checkbox.isChecked()
            },
            'cell_type': self.cell_type_combo.currentText(),
            'cell_radius': self.cell_radius.value(),
            'membrane_thickness': self.membrane_thickness.value(),
            'nucleus_radius': self.nucleus_radius.value(),
            'er_density': self.er_density.value(),  # Add this line
            'pixel_size': (self.pixel_size_x.value(), self.pixel_size_y.value(), self.pixel_size_z.value()),
            'mitochondria': {
                'count': self.mito_count.value(),
                'size_range': (self.mito_size_min.value(), self.mito_size_max.value())
            },
            'cytoskeleton': {
                'actin_density': self.actin_density.value(),
                'microtubule_density': self.microtubule_density.value()
            },
            'calcium_signal': {
                'type': self.calcium_combo.currentText(),
                'intensity': self.calcium_intensity.value(),
                'duration': self.calcium_duration.value()
            },
            'neuron': {
                'soma_radius': self.soma_radius.value() if self.cell_type_combo.currentText() == 'neuron' else None,
                'axon_length': self.axon_length.value() if self.cell_type_combo.currentText() == 'neuron' else None,
                'axon_width': self.axon_width.value() if self.cell_type_combo.currentText() == 'neuron' else None,
                'num_dendrites': self.num_dendrites.value() if self.cell_type_combo.currentText() == 'neuron' else None,
                'dendrite_length': self.dendrite_length.value() if self.cell_type_combo.currentText() == 'neuron' else None
            }
        }
        self.simulationRequested.emit(params)


class BiologicalSimulationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Biological Simulation")
        self.simulation_widget = BiologicalSimulationWidget()
        self.setCentralWidget(self.simulation_widget)
        self.resize(400, 300)  # Set an initial size for the window


class VolumeProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def apply_threshold(self, data, threshold):
        try:
            return np.where(data > threshold, data, 0)
        except Exception as e:
            self.logger.error(f"Error in applying threshold: {str(e)}")
            return data

    def apply_gaussian_filter(self, data, sigma):
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(data, sigma)
        except ImportError:
            self.logger.error("SciPy not installed. Cannot apply Gaussian filter.")
            return data
        except Exception as e:
            self.logger.error(f"Error in applying Gaussian filter: {str(e)}")
            return data

    def calculate_statistics(self, data):
        try:
            stats = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
            self.logger.info(f"Statistics calculated: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Error in calculating statistics: {str(e)}")
            return {}



class BlobAnalysisDialog(QDialog):
    def __init__(self, blob_analyzer, parent=None):
        super().__init__(parent)
        self.blob_analyzer = blob_analyzer
        self.setWindowTitle("Blob Analysis Results")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Add time point selection
        self.timeComboBox = QComboBox()
        self.timeComboBox.addItems([str(t) for t in self.blob_analyzer.time_points])
        self.timeComboBox.currentIndexChanged.connect(self.updateAnalysis)
        layout.addWidget(self.timeComboBox)

        self.tabWidget = QTabWidget()
        layout.addWidget(self.tabWidget)

        # Add a close button
        closeButton = QPushButton("Close")
        closeButton.clicked.connect(self.close)
        layout.addWidget(closeButton)

        self.setLayout(layout)

        self.updateAnalysis()

    def updateAnalysis(self):
        self.tabWidget.clear()
        time_point = int(self.timeComboBox.currentText())

        self.addDistanceAnalysisTab(time_point)
        self.addDensityAnalysisTab(time_point)
        self.addColocalizationTab(time_point)
        self.addBlobSizeTab(time_point)
        self.addIntensityAnalysisTab(time_point)
        self.add3DVisualizationTab(time_point)
        self.addStatsTab(time_point)

    def addDistanceAnalysisTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        all_distances, within_channel_distances, between_channel_distances = self.blob_analyzer.calculate_nearest_neighbor_distances(time_point)

        plt.figure(figsize=(10, 6))
        plt.hist(all_distances, bins=50, alpha=0.5, label='All Blobs')
        for ch, distances in within_channel_distances.items():
            plt.hist(distances, bins=50, alpha=0.5, label=f'Channel {ch}')
        plt.xlabel('Nearest Neighbor Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Nearest Neighbor Distance Distribution (Time: {time_point})')
        canvas = FigureCanvas(plt.gcf())
        layout.addWidget(canvas)

        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Distance Analysis")

        plt.close()

    def addDensityAnalysisTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        overall_density, channel_densities = self.blob_analyzer.calculate_blob_density((30, 100, 100), time_point)

        textEdit = QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.append(f"Time Point: {time_point}")
        textEdit.append(f"Overall Blob Density: {overall_density:.6f} blobs/unit^3")
        for ch, density in channel_densities.items():
            textEdit.append(f"Channel {ch} Density: {density:.6f} blobs/unit^3")

        layout.addWidget(textEdit)
        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Density Analysis")

    def addBlobSizeTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        blob_sizes = self.blob_analyzer.calculate_blob_sizes(time_point)

        plt.figure(figsize=(10, 6))
        for ch, sizes in blob_sizes.items():
            plt.hist(sizes, bins=50, alpha=0.5, label=f'Channel {ch}')
        plt.xlabel('Blob Size')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Blob Size Distribution (Time: {time_point})')
        canvas = FigureCanvas(plt.gcf())
        layout.addWidget(canvas)

        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Blob Size Analysis")

        plt.close()

    def addStatsTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        textEdit = QTextEdit()
        textEdit.setReadOnly(True)

        all_distances, within_channel_distances, _ = self.blob_analyzer.calculate_nearest_neighbor_distances(time_point)
        overall_density, channel_densities = self.blob_analyzer.calculate_blob_density((30, 100, 100), time_point)
        blob_sizes = self.blob_analyzer.calculate_blob_sizes(time_point)

        textEdit.append(f"Statistics for Time Point: {time_point}")
        textEdit.append("\nOverall Statistics:")
        time_blobs = self.blob_analyzer.blobs[self.blob_analyzer.blobs[:, 5] == time_point]
        textEdit.append(f"Total number of blobs: {len(time_blobs)}")
        textEdit.append(f"Overall blob density: {overall_density:.6f} blobs/unit^3")
        if len(all_distances) > 0:
            textEdit.append(f"Mean nearest neighbor distance: {np.mean(all_distances):.2f}")
            textEdit.append(f"Median nearest neighbor distance: {np.median(all_distances):.2f}")
        else:
            textEdit.append("Not enough blobs to calculate nearest neighbor distances.")

        for ch in self.blob_analyzer.channels:
            textEdit.append(f"\nChannel {ch} Statistics:")
            channel_blobs = time_blobs[time_blobs[:, 4] == ch]
            textEdit.append(f"Number of blobs: {len(channel_blobs)}")
            textEdit.append(f"Blob density: {channel_densities[ch]:.6f} blobs/unit^3")
            if len(blob_sizes[ch]) > 0:
                textEdit.append(f"Mean blob size: {np.mean(blob_sizes[ch]):.2f}")
                textEdit.append(f"Median blob size: {np.median(blob_sizes[ch]):.2f}")
            else:
                textEdit.append("No blobs detected in this channel.")
            if ch in within_channel_distances and len(within_channel_distances[ch]) > 0:
                textEdit.append(f"Mean nearest neighbor distance: {np.mean(within_channel_distances[ch]):.2f}")
                textEdit.append(f"Median nearest neighbor distance: {np.median(within_channel_distances[ch]):.2f}")
            else:
                textEdit.append("Not enough blobs to calculate nearest neighbor distances.")

        layout.addWidget(textEdit)
        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Statistics")

    def addIntensityAnalysisTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        intensities = self.blob_analyzer.calculate_blob_intensities(time_point)
        sizes = self.blob_analyzer.calculate_blob_sizes(time_point)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for ch in self.blob_analyzer.channels:
            ax1.hist(intensities[ch], bins=50, alpha=0.5, label=f'Channel {ch}')
        ax1.set_xlabel('Intensity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Blob Intensity Distribution')
        ax1.legend()

        for ch in self.blob_analyzer.channels:
            ax2.scatter(sizes[ch], intensities[ch], alpha=0.5, label=f'Channel {ch}')
        ax2.set_xlabel('Blob Size')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Blob Size vs Intensity')
        ax2.legend()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Intensity Analysis")

        plt.close()

    def add3DVisualizationTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        time_blobs = self.blob_analyzer.blobs[self.blob_analyzer.blobs[:, 5] == time_point]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        for ch in self.blob_analyzer.channels:
            channel_blobs = time_blobs[time_blobs[:, 4] == ch]
            ax.scatter(channel_blobs[:, 0], channel_blobs[:, 1], channel_blobs[:, 2],
                       s=channel_blobs[:, 3]*10, alpha=0.5, label=f'Channel {ch}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Blob Positions (Time: {time_point})')
        ax.legend()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "3D Visualization")

        plt.close()

    def addColocalizationTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        basic_coloc = self.blob_analyzer.calculate_colocalization(distance_threshold=5, time_point=time_point)
        advanced_coloc = self.blob_analyzer.calculate_advanced_colocalization(time_point)

        textEdit = QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.append(f"Time Point: {time_point}")
        textEdit.append("\nBasic Colocalization:")
        for (ch1, ch2), coloc in basic_coloc.items():
            textEdit.append(f"Channels {ch1} and {ch2}: {coloc:.2%}")

        textEdit.append("\nAdvanced Colocalization:")
        for (ch1, ch2), results in advanced_coloc.items():
            textEdit.append(f"Channels {ch1} and {ch2}:")
            pearson = results['pearson']
            textEdit.append(f"  Pearson's coefficient: {pearson:.4f}" if not np.isnan(pearson) else "  Pearson's coefficient: N/A")
            textEdit.append(f"  Manders' M1: {results['manders_m1']:.4f}")
            textEdit.append(f"  Manders' M2: {results['manders_m2']:.4f}")

        layout.addWidget(textEdit)
        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Colocalization")


class BlobResultsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blob Detection Results")
        self.layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

    def update_results(self, blobs):
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["X", "Y", "Z", "Size", "Channel", "Time"])
        self.table.setRowCount(len(blobs))

        for i, blob in enumerate(blobs):
            y, x, z, r, channel, t = blob
            self.table.setItem(i, 0, QTableWidgetItem(f"{x:.2f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{y:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{z:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{r:.2f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{int(channel)}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{int(t)}"))

class ROI3D(gl.GLMeshItem):
    sigRegionChanged = pyqtSignal(object)

    def __init__(self, size=(10, 10, 10), color=(1, 1, 1, 0.3)):
        verts, faces = self.create_cube(size)
        super().__init__(vertexes=verts, faces=faces, smooth=False, drawEdges=True, edgeColor=color)
        self.size = size
        self.setColor(color)

    @staticmethod
    def create_cube(size):
        x, y, z = size
        verts = np.array([
            [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
            [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 4, 5],
            [1, 2, 5], [2, 5, 6], [2, 3, 6], [3, 6, 7],
            [3, 0, 7], [0, 4, 7], [4, 5, 6], [4, 6, 7]
        ])
        return verts, faces

    def setPosition(self, pos):
        self.resetTransform()
        self.translate(*pos)
        self.sigRegionChanged.emit(self)

class TimeSeriesDialog(QDialog):
    def __init__(self, blob_analyzer, parent=None):
        super().__init__(parent)
        self.blob_analyzer = blob_analyzer
        self.setWindowTitle("Time Series Analysis")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        plot_widget = pg.PlotWidget()
        layout.addWidget(plot_widget)

        time_points = np.unique(self.blob_analyzer.blobs[:, 5])
        channels = np.unique(self.blob_analyzer.blobs[:, 4])

        for channel in channels:
            blob_counts = [np.sum((self.blob_analyzer.blobs[:, 5] == t) & (self.blob_analyzer.blobs[:, 4] == channel))
                           for t in time_points]
            plot_widget.plot(time_points, blob_counts, pen=(int(channel), len(channels)), name=f'Channel {int(channel)}')

        plot_widget.setLabel('left', "Number of Blobs")
        plot_widget.setLabel('bottom', "Time Point")
        plot_widget.addLegend()

        self.setLayout(layout)


class Visualizer3D:
    def __init__(self, parent):
        self.parent = parent
        self.glView = None
        self.blobGLView = None
        self.data_items = []
        self.blob_items = []
        self.main_slice_marker_items = []
        self.slice_marker_items = []

    def create_3d_view(self):
        self.glView = gl.GLViewWidget()
        self.glView.setCameraPosition(distance=50, elevation=30, azimuth=45)
        self.glView.opts['backgroundColor'] = pg.mkColor(20, 20, 20)  # Dark background

        # Add a grid to the view
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        self.glView.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        self.glView.addItem(gy)
        gz = gl.GLGridItem()
        self.glView.addItem(gz)

        self.glView.opts['fov'] = 60
        self.glView.opts['elevation'] = 30
        self.glView.opts['azimuth'] = 45

        return self.glView

    def create_blob_view(self):
        self.blobGLView = gl.GLViewWidget()

        # Add a grid to the view
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        self.blobGLView.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        self.blobGLView.addItem(gy)
        gz = gl.GLGridItem()
        self.blobGLView.addItem(gz)

        return self.blobGLView

    def clear_3d_visualization(self):
        for item in self.data_items:
            self.glView.removeItem(item)
        self.data_items.clear()

    def update_3d_visualization(self, data, time_point, threshold, render_mode):
        self.clear_3d_visualization()

        for c in range(data.shape[1]):
            if self.parent.isChannelVisible(c):
                try:
                    self.visualize_channel(data, c, time_point, threshold, render_mode)
                except Exception as e:
                    self.parent.logger.error(f"Error visualizing channel {c}: {str(e)}")
                    self.parent.logger.error(f"Traceback: {traceback.format_exc()}")

        self.glView.update()

    def visualize_channel(self, data, channel, time, threshold, render_mode):
        volume_data = data[time, channel]
        opacity = self.parent.getChannelOpacity(channel)
        color = self.parent.getChannelColor(channel)
        color = color[:3]  # Remove alpha if present

        if render_mode == 'points':
            self.render_points(volume_data, threshold, color, opacity)
        else:
            self.render_mesh(volume_data, threshold, color, opacity, render_mode)

    def render_points(self, volume_data, threshold, color, opacity):
        z, y, x = np.where(volume_data > threshold)
        pos = np.column_stack((x, y, z))

        if len(pos) > 0:
            # Create colors array with 4 channels (RGBA)
            colors = np.zeros((len(pos), 4))
            colors[:, :3] = color  # Set RGB values

            # Calculate alpha based on intensity
            intensity = (volume_data[z, y, x] - volume_data.min()) / (volume_data.max() - volume_data.min())
            colors[:, 3] = opacity * intensity  # Set alpha values

            scatter = gl.GLScatterPlotItem(pos=pos, color=colors, size=self.parent.pointSizeSpinBox.value())
            self.glView.addItem(scatter)
            self.data_items.append(scatter)

    def render_mesh(self, volume_data, threshold, color, opacity, render_mode):
        verts, faces = self.parent.marchingCubes(volume_data, threshold)
        if len(verts) > 0 and len(faces) > 0:
            mesh = gl.GLMeshItem(vertexes=verts, faces=faces, smooth=True, drawEdges=render_mode=='wireframe')
            mesh.setColor(color + (opacity,))

            if render_mode == 'wireframe':
                mesh.setColor(color + (opacity*0.1,))
                mesh = gl.GLMeshItem(vertexes=verts, faces=faces, smooth=True, drawEdges=True,
                                     edgeColor=color + (opacity,))

            self.glView.addItem(mesh)
            self.data_items.append(mesh)


    def clear_slice_markers(self):
        for item in self.slice_marker_items:
            try:
                self.blobGLView.removeItem(item)
            except:
                pass

        for item in self.main_slice_marker_items:
            try:
                self.glView.removeItem(item)
            except:
                pass

        self.slice_marker_items.clear()
        self.main_slice_marker_items.clear()

    def create_slice_markers(self, x_slice, y_slice, z_slice, width, height, depth):
        x_marker = gl.GLLinePlotItem(pos=np.array([[x_slice, 0, 0], [x_slice, height, 0], [x_slice, height, depth], [x_slice, 0, depth]]),
                                     color=(1, 0, 0, 1), width=2, mode='line_strip')
        y_marker = gl.GLLinePlotItem(pos=np.array([[0, y_slice, 0], [width, y_slice, 0], [width, y_slice, depth], [0, y_slice, depth]]),
                                     color=(0, 1, 0, 1), width=2, mode='line_strip')
        z_marker = gl.GLLinePlotItem(pos=np.array([[0, 0, z_slice], [width, 0, z_slice], [width, height, z_slice], [0, height, z_slice]]),
                                     color=(0, 0, 1, 1), width=2, mode='line_strip')

        self.glView.addItem(x_marker)
        self.glView.addItem(y_marker)
        self.glView.addItem(z_marker)

        x_marker_vis = gl.GLLinePlotItem(pos=np.array([[x_slice, 0, 0], [x_slice, height, 0], [x_slice, height, depth], [x_slice, 0, depth]]),
                                         color=(1, 0, 0, 1), width=2, mode='line_strip')
        y_marker_vis = gl.GLLinePlotItem(pos=np.array([[0, y_slice, 0], [width, y_slice, 0], [width, y_slice, depth], [0, y_slice, depth]]),
                                         color=(0, 1, 0, 1), width=2, mode='line_strip')
        z_marker_vis = gl.GLLinePlotItem(pos=np.array([[0, 0, z_slice], [width, 0, z_slice], [width, height, z_slice], [0, height, z_slice]]),
                                         color=(0, 0, 1, 1), width=2, mode='line_strip')

        self.blobGLView.addItem(x_marker_vis)
        self.blobGLView.addItem(y_marker_vis)
        self.blobGLView.addItem(z_marker_vis)

        self.slice_marker_items.extend([x_marker_vis, y_marker_vis, z_marker_vis])
        self.main_slice_marker_items.extend([x_marker, y_marker, z_marker])

        # Add other visualization methods here...

class ShapeSimulationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shape Simulation Setup")
        self.layout = QVBoxLayout(self)

        self.shape_list = QListWidget()
        self.layout.addWidget(self.shape_list)

        self.add_shape_button = QPushButton("Add Shape")
        self.add_shape_button.clicked.connect(self.add_shape)
        self.layout.addWidget(self.add_shape_button)

        self.resolution = QSpinBox()
        self.resolution.setRange(5, 50)
        self.resolution.setValue(10)
        self.layout.addWidget(QLabel("Resolution:"))
        self.layout.addWidget(self.resolution)

        self.interaction_combo = QComboBox()
        self.interaction_combo.addItems(["None", "Attraction", "Repulsion"])
        self.layout.addWidget(self.interaction_combo)

        self.num_steps = QSpinBox()
        self.num_steps.setRange(1, 1000)
        self.num_steps.setValue(100)
        self.layout.addWidget(QLabel("Number of steps:"))
        self.layout.addWidget(self.num_steps)

        self.dt = QDoubleSpinBox()
        self.dt.setRange(0.01, 1.0)
        self.dt.setValue(0.1)
        self.layout.addWidget(QLabel("Time step (dt):"))
        self.layout.addWidget(self.dt)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.accept)
        self.layout.addWidget(self.run_button)

    def add_shape(self):
        dialog = ShapeDialog(self)
        if dialog.exec_():
            self.shape_list.addItem(str(dialog.get_shape_params()))

    def get_params(self):
        shapes = [eval(self.shape_list.item(i).text()) for i in range(self.shape_list.count())]
        return {
            'shapes': shapes,
            'interactions': [{'type': self.interaction_combo.currentText().lower()}],
            'num_steps': self.num_steps.value(),
            'dt': self.dt.value(),
            'resolution': self.resolution.value()
        }

class ShapeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Shape")
        self.layout = QFormLayout(self)

        self.shape_type = QComboBox()
        self.shape_type.addItems(["Sphere", "Cube"])
        self.layout.addRow("Shape Type:", self.shape_type)

        self.position = [QDoubleSpinBox() for _ in range(3)]
        for pos in self.position:
            pos.setRange(-50, 50)
            pos.setDecimals(2)  # Allow for decimal places
        self.layout.addRow("Position:", self.create_widget_group(self.position))


        self.size = QSpinBox()
        self.size.setRange(1, 20)
        self.layout.addRow("Size:", self.size)

        self.color = [QSpinBox() for _ in range(3)]
        for c in self.color:
            c.setRange(0, 255)
        self.layout.addRow("Color (RGB):", self.create_widget_group(self.color))

        self.movement = QComboBox()
        self.movement.addItems(["Random", "Linear"])
        self.layout.addRow("Movement:", self.movement)

        self.speed = QDoubleSpinBox()
        self.speed.setRange(0, 10)
        self.layout.addRow("Speed:", self.speed)

        self.accept_button = QPushButton("Add")
        self.accept_button.clicked.connect(self.accept)
        self.layout.addRow(self.accept_button)

    def create_widget_group(self, widgets):
        group = QWidget()
        layout = QHBoxLayout(group)
        for widget in widgets:
            layout.addWidget(widget)
        return group

    def get_shape_params(self):
        return {
            'type': self.shape_type.currentText().lower(),
            'position': [pos.value() for pos in self.position],
            'size': self.size.value(),
            'color': [c.value() for c in self.color],
            'movement': self.movement.currentText().lower(),
            'speed': self.speed.value(),
            'velocity': [self.speed.value(), 0, 0] if self.movement.currentText().lower() == 'linear' else None
        }

######################################################################################################
######################################################################################################
'''                                   MAIN LIGHTSHEETVIEWER CLASS                                  '''
######################################################################################################
######################################################################################################

class LightsheetViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initLogging()
        self.initColors()
        self.data_manager = DataManager()
        self.visualization_manager = VisualizationManager(self)
        self.visualization_manager.channel_colors = self.channel_colors
        self.ui_manager = UIManager(self)
        self.volume_processor = VolumeProcessor()
        self.data = None
        self.lastPos = None
        self.data_manager = DataManager()
        self.initSimulator()
        self.raw_data = None
        self.data_properties = None
        self.raw_data_viewer = None
        self.image_processor = ImageProcessor()
        self.createRawDataMenu()

        # Initialize the Visualizer3D
        self.visualizer = Visualizer3D(self)

        self.initUI()
        self.generateData()
        self.initTimer()
        self.blob_results_dialog = BlobResultsDialog(self)
        self.showBlobResultsButton.setVisible(False)
        self.biological_simulation_window = None
        self.toggleDownsamplingControls()  # Set initial state of downsampling controls

        # Initialize empty lists for items that are now managed by Visualizer3D
        self.data_items = []
        self.blob_items = []
        self.main_slice_marker_items = []
        self.slice_marker_items = []


    def initColors(self):
        self.channel_colors = [
            QColor(255, 0, 0),    # Red
            QColor(0, 255, 0),    # Green
            QColor(0, 0, 255),    # Blue
            QColor(255, 255, 0),  # Yellow
            QColor(255, 0, 255),  # Magenta
            QColor(0, 255, 255),  # Cyan
            QColor(128, 128, 128),  # Gray
            QColor(255, 128, 0),  # Orange
            QColor(128, 0, 128),  # Purple
            QColor(0, 128, 128),  # Teal
        ]


    def initLogging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def initTimer(self):
        self.playbackTimer = QTimer(self)
        self.playbackTimer.timeout.connect(self.advanceTimePoint)

    def initUI(self):
        self.setWindowTitle('Lightsheet Microscopy Viewer')
        self.setGeometry(100, 100, 1600, 900)
        self.setCentralWidget(None)

        self.ui_manager.setup_ui()
        self.organizeDocks()
        self.createMenuBar()

        self.connectViewEvents()
        self.check_view_state()


    def organizeDocks(self):
        self.setDockOptions(QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)

        # Stack XY, XZ, and YZ views vertically on the left
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockXY)
        self.splitDockWidget(self.dockXY, self.dockXZ, Qt.Vertical)
        self.splitDockWidget(self.dockXZ, self.dockYZ, Qt.Vertical)

        # Add 3D view to the right of the 2D views
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock3D)

        # Add control docks to the far right
        self.addDockWidget(Qt.RightDockWidgetArea, self.ui_manager.dockChannelControls)  # Add this line
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockDataGeneration)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockVisualizationControl)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockPlaybackControl)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockBlobDetection)

        # Set the 3D view and control docks side by side
        self.splitDockWidget(self.dock3D, self.ui_manager.dockChannelControls, Qt.Horizontal)  # Modified this line
        self.tabifyDockWidget(self.ui_manager.dockChannelControls, self.dockDataGeneration)  # Modified this line
        self.tabifyDockWidget(self.dockDataGeneration, self.dockVisualizationControl)
        self.tabifyDockWidget(self.dockVisualizationControl, self.dockPlaybackControl)
        self.tabifyDockWidget(self.dockPlaybackControl, self.dockBlobDetection)

        # Add blob visualization dock below the 3D view
        self.splitDockWidget(self.dock3D, self.dockBlobVisualization, Qt.Vertical)

        # Adjust dock sizes
        self.resizeDocks([self.dockXY, self.dockXZ, self.dockYZ], [200, 200, 200], Qt.Vertical)
        self.resizeDocks([self.dock3D, self.ui_manager.dockChannelControls], [800, 300], Qt.Horizontal)  # Modified this line


    def initSimulator(self):
        self.biological_simulator = BiologicalSimulator(size=(30, 100, 100), num_time_points=10)
        #TODO! Add the enhanced biological Simulator as an alternative option for simulating multiple cells and cell interactions
        self.enhanced_biological_simulator = None
        self.current_simulation_type = 'original'

    def initDataGenerator(self):
        self.data_generator = DataGenerator()


    def runBiologicalSimulation(self, params):
        try:
            self.data = None  # Reset data at the start of simulation
            self.current_simulation_type = 'original'
            soma_center = tuple(s // 2 for s in self.biological_simulator.size)
            cell_type = params['cell_type']
            cell_shape, cell_interior, cell_membrane = self.biological_simulator.generate_cell_shape(
                cell_type,
                self.biological_simulator.size,
                params['pixel_size'],
                membrane_thickness=params['membrane_thickness'],
                cell_radius=params['cell_radius']
            )
            self.generateCellularStructures(params, cell_shape, cell_interior, cell_membrane, soma_center)
            self.simulateProteinDynamics(params)
            self.simulateCalciumSignal(params)

            self.updateUIForNewData()
            self.updateViews()
            self.create3DVisualization()

        except Exception as e:
            self.handleSimulationError(e)

    def generateCellShape(self, params, soma_center, cell_type):
        if cell_type == 'neuron':
            return self.biological_simulator.generate_cell_shape(
                cell_type,
                self.biological_simulator.size,
                params['pixel_size'],
                membrane_thickness=params['membrane_thickness'],
                cell_radius=params['cell_radius'],
                soma_radius=params['neuron']['soma_radius'],
                axon_length=params['neuron']['axon_length'],
                axon_width=params['neuron']['axon_width'],
                num_dendrites=params['neuron']['num_dendrites'],
                dendrite_length=params['neuron']['dendrite_length']
            )
        else:
            return self.biological_simulator.generate_cell_shape(
                cell_type,
                self.biological_simulator.size,
                params['pixel_size'],
                membrane_thickness=params['membrane_thickness'],
                cell_radius=params['cell_radius']
            )

    def generateCellularStructures(self, params, cell_shape, cell_interior, cell_membrane, soma_center):
        cellular_structures = params.get('cellular_structures', {})

        if cellular_structures.get('cell_membrane', False):
            self.generateCellMembrane(cell_membrane)

        if cellular_structures.get('nucleus', False):
            self.generateNucleus(params, cell_interior, soma_center)

        if cellular_structures.get('er', False):
            self.generateER(params, cell_shape, soma_center)

        if cellular_structures.get('mitochondria', False):
            self.generateMitochondria(params)

        if cellular_structures.get('cytoskeleton', False):
            self.generateCytoskeleton(params)

    def generateCellMembrane(self, cell_membrane):
        membrane_timeseries = np.repeat(cell_membrane[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
        self.addDataChannel(membrane_timeseries, "Cell Membrane")

    def generateNucleus(self, params, cell_interior, soma_center):
        nucleus_data, nucleus_center = self.biological_simulator.generate_nucleus(
            cell_interior,
            soma_center,
            params['nucleus_radius'],
            pixel_size=params['pixel_size']
        )
        nucleus_timeseries = np.repeat(nucleus_data[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
        self.addDataChannel(nucleus_timeseries, "Nucleus")

    def generateER(self, params, cell_shape, soma_center):
        er_data = self.biological_simulator.generate_er(
            cell_shape,
            soma_center,
            params['nucleus_radius'],
            params['er_density'],
            params['pixel_size']
        )
        er_timeseries = np.repeat(er_data[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
        self.addDataChannel(er_timeseries, "ER")

    def generateMitochondria(self, params):
        mito_count = params['mitochondria'].get('count', 50)
        mito_size_range = params['mitochondria'].get('size_range', (3, 8))
        mitochondria = self.biological_simulator.generate_mitochondria(mito_count, mito_size_range)
        mitochondria_timeseries = np.repeat(mitochondria[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
        self.addDataChannel(mitochondria_timeseries, "Mitochondria")

    def generateCytoskeleton(self, params):
        actin_density = params['cytoskeleton'].get('actin_density', 0.05)
        microtubule_density = params['cytoskeleton'].get('microtubule_density', 0.02)
        actin, microtubules = self.biological_simulator.generate_cytoskeleton(actin_density, microtubule_density)
        actin_timeseries = np.repeat(actin[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
        microtubules_timeseries = np.repeat(microtubules[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
        self.addDataChannel(actin_timeseries, "Actin")
        self.addDataChannel(microtubules_timeseries, "Microtubules")

    def simulateProteinDynamics(self, params):
        if params['protein_diffusion']['enabled']:
            self.simulateProteinDiffusion(params)
        if params['active_transport']['enabled']:
            self.simulateActiveTransport(params)

    def simulateProteinDiffusion(self, params):
        diffusion_coefficient = params['protein_diffusion']['coefficient']
        initial_concentration = np.zeros(self.biological_simulator.size)
        initial_concentration[self.biological_simulator.size[0]//2,
                              self.biological_simulator.size[1]//2,
                              self.biological_simulator.size[2]//2] = 1.0  # Point source in the center
        diffusion_data = self.biological_simulator.simulate_protein_diffusion(
            diffusion_coefficient,
            initial_concentration
        )
        self.addDataChannel(diffusion_data[:, np.newaxis, :, :, :], "Protein Diffusion")

    def simulateActiveTransport(self, params):
        velocity = params['active_transport'].get('velocity', (1, 1, 1))
        use_microtubules = params['active_transport'].get('use_microtubules', True)
        initial_cargo = np.zeros(self.biological_simulator.size)
        initial_cargo[self.biological_simulator.size[0]//2, self.biological_simulator.size[1]//2, self.biological_simulator.size[2]//2] = 1.0
        transport_data = self.biological_simulator.simulate_active_transport(velocity, initial_cargo, use_microtubules)
        self.addDataChannel(transport_data[:, np.newaxis, :, :, :], "Active Transport")

    def simulateCalciumSignal(self, params):
        if params['calcium_signal']['type'] != 'None':
            calcium_signal_type = params['calcium_signal']['type']
            calcium_signal = self.biological_simulator.simulate_calcium_signal(calcium_signal_type.lower(), {})
            self.addDataChannel(calcium_signal[:, np.newaxis, :, :, :], "Calcium Signal")

    def addDataChannel(self, channel_data, channel_name):
        if self.data is None:
            self.data = channel_data
        else:
            self.data = np.concatenate((self.data, channel_data), axis=1)
        self.logger.info(f"Added {channel_name} data. New data shape: {self.data.shape}")

    def handleSimulationError(self, e):
        self.logger.error(f"Error in biological simulation: {str(e)}")
        self.logger.error(f"Full exception: {traceback.format_exc()}")
        QMessageBox.critical(self, "Simulation Error", f"An error occurred during simulation: {str(e)}")


    def updateMarkersFromSliders(self):
        self.updateSliceMarkers()
        self.create3DVisualization()  # This might be heavy, consider optimizing if performance is an issue


    def updateSliceMarkers(self):
        if not hasattr(self, 'data') or self.data is None:
            return

        self.visualization_manager.clear_slice_markers()

        if self.showSliceMarkersCheck.isChecked():
            _, _, depth, height, width = self.data.shape  # Assuming shape is (t, c, z, y, x)

            z_slice = int(self.imageViewXY.currentIndex)
            y_slice = int(self.imageViewXZ.currentIndex)
            x_slice = int(self.imageViewYZ.currentIndex)

            self.visualization_manager.create_slice_markers(x_slice, y_slice, z_slice, width, height, depth)

        self.glView.update()
        self.blobGLView.update()


    def create3DVisualization(self):
        try:
            t = self.timeSlider.value()
            threshold = self.thresholdSpinBox.value()
            render_mode = self.getRenderMode()

            # Apply downsampling if enabled
            if self.downsamplingCheckBox.isChecked():
                downsampled_data = self.downsampleData(self.data[t], self.downsamplingSpinBox.value())
            else:
                downsampled_data = self.data[t]

            self.visualization_manager.update_3d_visualization(downsampled_data, t, threshold, render_mode)
        except Exception as e:
            self.handle3DVisualizationError(e)


    def visualize_blobs(self, blobs):
        current_time = self.timeSlider.value()
        self.visualization_manager.visualize_blobs(blobs, current_time, self.channel_colors)

    def advanceTimePoint(self):
        current_time = self.timeSlider.value()
        if current_time < self.timeSlider.maximum():
            self.timeSlider.setValue(current_time + 1)
        elif self.loopCheckBox.isChecked():
            self.timeSlider.setValue(0)
        else:
            self.playbackTimer.stop()
            self.playPauseButton.setText("Play")

    def updateTimePoint(self, value):
        if self.data is not None:
            self.currentTimePoint = value
            self.updateViews()
            self.create3DVisualization()
            self.updateBlobVisualization()
        else:
            self.logger.warning("No data available to update time point")


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Adjust dock sizes to maintain aspect ratio if needed
        width = self.width()
        left_width = width // 3
        right_width = width - left_width

        # Resize the 2D view docks
        self.resizeDocks([self.dockXY, self.dockXZ, self.dockYZ], [left_width] * 3, Qt.Horizontal)

        # Resize the 3D view and control docks
        control_width = right_width // 4  # Allocate 1/4 of right side to controls
        self.resizeDocks([self.dock3D], [right_width - control_width], Qt.Horizontal)
        self.resizeDocks([self.dockDataGeneration, self.dockVisualizationControl, self.dockPlaybackControl],
                         [control_width] * 3, Qt.Horizontal)

    def createMenuBar(self):
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu('&File')

        loadAction = fileMenu.addAction('&Load Data')
        loadAction.triggered.connect(self.loadData)

        saveAction = fileMenu.addAction('&Save Data')
        saveAction.triggered.connect(self.saveData)

        importAction = fileMenu.addAction('&Import Microscope Data')
        importAction.triggered.connect(self.import_microscope_data)

        quitAction = fileMenu.addAction('&Quit')
        quitAction.triggered.connect(self.close)

        viewMenu = menuBar.addMenu('&View')

        for dock in [self.dockXY, self.dockXZ, self.dockYZ, self.dock3D,
                     self.ui_manager.dockChannelControls,  # Add this line
                     self.dockDataGeneration, self.dockVisualizationControl,
                     self.dockPlaybackControl, self.dockBlobDetection]:
            viewMenu.addAction(dock.toggleViewAction())


        analysisMenu = menuBar.addMenu('&Analysis')
        timeSeriesAction = analysisMenu.addAction('Time Series Analysis')
        timeSeriesAction.triggered.connect(self.showTimeSeriesAnalysis)

        # Add a new menu item for the Biological Simulation window
        simulationMenu = menuBar.addMenu('&Simulation')
        self.showBioSimAction = QAction('Biological Simulation', self, checkable=True)
        self.showBioSimAction.triggered.connect(self.toggleBiologicalSimulationWindow)
        simulationMenu.addAction(self.showBioSimAction)

        # Add this new action
        self.showEnhancedBioSimAction = QAction('Enhanced Biological Simulation', self, checkable=True)
        self.showEnhancedBioSimAction.triggered.connect(self.toggleEnhancedBiologicalSimulationWindow)

        self.shapeSimAction = simulationMenu.addAction('Run Shape Simulation')
        self.shapeSimAction.triggered.connect(self.showShapeSimulationDialog)


        simulationMenu.addAction(self.showEnhancedBioSimAction)

    def showShapeSimulationDialog(self):
        dialog = ShapeSimulationDialog(self)
        if dialog.exec_():
            params = dialog.get_params()
            self.runShapeSimulation(params)


    def toggleEnhancedBiologicalSimulationWindow(self, checked):
        if checked:
            if not hasattr(self, 'enhanced_biological_simulation_window'):
                self.enhanced_biological_simulation_window = QMainWindow(self)
                self.enhanced_biological_simulation_window.setWindowTitle("Enhanced Biological Simulation")
                simulation_widget = EnhancedBiologicalSimulationWidget()
                simulation_widget.simulationRequested.connect(self.runEnhancedBiologicalSimulation)
                self.enhanced_biological_simulation_window.setCentralWidget(simulation_widget)
            self.enhanced_biological_simulation_window.show()
        else:
            if hasattr(self, 'enhanced_biological_simulation_window'):
                self.enhanced_biological_simulation_window.hide()

    def toggleBiologicalSimulationWindow(self, checked):
        if checked:
            if self.biological_simulation_window is None:
                self.biological_simulation_window = QMainWindow(self)
                self.biological_simulation_window.setWindowTitle("Biological Simulation")
                simulation_widget = BiologicalSimulationWidget()
                simulation_widget.simulationRequested.connect(self.runBiologicalSimulation)
                self.biological_simulation_window.setCentralWidget(simulation_widget)
            self.biological_simulation_window.show()
        else:
            if self.biological_simulation_window:
                self.biological_simulation_window.hide()


    def generateData(self):
        try:
            params = self.getDataGenerationParams()
            self.data = self.data_manager.generate_data(params)
            self.updateUIForNewData()
            self.updateViews()
            self.create3DVisualization()
            self.autoScaleViews()
            self.updateRawDataViewer()  # Add this line
            self.logger.info("Data generated and visualized successfully")
        except Exception as e:
            self.handleDataGenerationError(e)

    def getDataGenerationParams(self):
        params = {
            'num_volumes': self.numVolumesSpinBox.value(),
            'num_channels': 2,  # Adjust if you allow variable number of channels
            'num_blobs': self.numBlobsSpinBox.value(),
            'noise_level': self.noiseLevelSpinBox.value(),
            'movement_speed': self.movementSpeedSpinBox.value(),
            'structured_data': self.structuredDataCheck.isChecked(),
            'size': (30, 100, 100),  # Adjust if you make this configurable
        }

        if params['structured_data']:
            params['channel_ranges'] = []
            for widgets in self.ui_manager.channelRangeWidgets:
                xMin, xMax, yMin, yMax, zMin, zMax = [w.value() for w in widgets]
                params['channel_ranges'].append({
                    'x': (xMin, xMax),
                    'y': (yMin, yMax),
                    'z': (zMin, zMax)
                })

        return params


    def handleDataGenerationError(self, e):
        self.logger.error(f"Error in data generation: {str(e)}")
        self.logger.error(f"Error type: {type(e).__name__}")
        self.logger.error(f"Error args: {e.args}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        QMessageBox.critical(self, "Error", f"Failed to generate data: {str(e)}")


    def updateViews(self):
        if self.data is None:
            self.logger.warning("No data to update views")
            return
        t = self.timeSlider.value()
        threshold = self.thresholdSpinBox.value()

        self.logger.debug(f"Updating views for time point {t}")
        self.logger.debug(f"Data shape: {self.data.shape}")

        combined_xy, combined_xz, combined_yz = self.visualization_manager.update_views(
            self.data, t, threshold, self.ui_manager.channel_controls)

        self.imageViewXY.setImage(combined_xy, autoLevels=False, levels=[0, 1])
        self.imageViewXZ.setImage(combined_xz, autoLevels=False, levels=[0, 1])
        self.imageViewYZ.setImage(combined_yz, autoLevels=False, levels=[0, 1])

        self.updateSliceMarkers()


    def isChannelVisible(self, channel):
        return (channel < len(self.ui_manager.channel_controls) and
                self.ui_manager.channel_controls[channel][0].isChecked())


    def hasDegenerateTriangles(self, verts, faces):
        # Check if any triangle has zero area
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.sqrt((cross**2).sum(axis=1))
        return np.any(areas < 1e-10)


    def calculateNormals(self, verts, faces):
        norm = np.zeros(verts.shape, dtype=verts.dtype)
        tris = verts[faces]
        n = np.cross(tris[::,1] - tris[::,0], tris[::,2] - tris[::,0])
        for i in range(3):
            norm[faces[:,i]] += n
        norm = norm / np.linalg.norm(norm, axis=1)[:, np.newaxis]
        return norm

    def filterDegenerateTriangles(self, verts, faces):
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.sqrt((cross**2).sum(axis=1))
        valid_faces = faces[areas > 1e-6]  # Increased threshold
        self.logger.info(f"Filtered out {len(faces) - len(valid_faces)} degenerate triangles out of {len(faces)}")
        return valid_faces

    def createEdgesFromFaces(self, faces):
        edges = set()
        for face in faces:
            for i in range(3):
                edge = (face[i], face[(i+1)%3])
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])
                edges.add(edge)
        return np.array(list(edges))


    def logDataStatistics(self, volume_data):
        self.logger.info(f"Data shape: {volume_data.shape}")
        self.logger.info(f"Data range: {volume_data.min()} to {volume_data.max()}")
        self.logger.info(f"Data mean: {volume_data.mean()}")
        self.logger.info(f"Data std: {volume_data.std()}")
        self.logger.info(f"Number of non-zero voxels: {np.count_nonzero(volume_data)}")


    def getChannelOpacity(self, channel):
        return self.ui_manager.channel_controls[channel][1].value() / 100


    def getRenderMode(self):
        return self.renderModeCombo.currentText().lower()

    def handle3DVisualizationError(self, e):
        self.logger.error(f"Error in 3D visualization: {str(e)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        QMessageBox.critical(self, "Error", f"Failed to create 3D visualization: {str(e)}")

    def visualize_data_distribution(self):
        t = self.timeSlider.value()
        num_channels, depth, height, width = self.data.shape[1:]

        for c in range(min(num_channels, 3)):
            volume_data = self.data[t, c]
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(np.max(volume_data, axis=0))
            plt.title(f'Channel {c} - XY Max Projection')
            plt.subplot(132)
            plt.imshow(np.max(volume_data, axis=1))
            plt.title(f'Channel {c} - XZ Max Projection')
            plt.subplot(133)
            plt.imshow(np.max(volume_data, axis=2))
            plt.title(f'Channel {c} - YZ Max Projection')
            plt.colorbar()
            plt.show()

        plt.figure()
        plt.hist(self.data[t].ravel(), bins=100)
        plt.title('Data Histogram')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.show()

    def updateVisualization(self):
        self.create3DVisualization()

    def updateScatterPointSize(self, value):
        self.updateViews()
        self.create3DVisualization()

    def updateChannelVisibility(self):
        self.updateViews()
        self.create3DVisualization()

    def updateChannelOpacity(self):
        self.updateViews()
        self.create3DVisualization()

    def getColorMap(self):
        cmap_name = self.colorMapCombo.currentText().lower()
        if cmap_name == "grayscale":
            return pg.ColorMap(pos=[0.0, 1.0], color=[(0, 0, 0, 255), (255, 255, 255, 255)])
        else:
            return pg.colormap.get(cmap_name)

    def triangulate_points(self, points):
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        return tri.simplices

    def updateRenderMode(self):
        self.create3DVisualization()

    def updateColorMap(self):
        self.create3DVisualization()

    def toggleSliceMarkers(self, state):
        if state == Qt.Checked:
            self.updateSliceMarkers()
        else:
            for attr in ['x_marker', 'y_marker', 'z_marker']:
                if hasattr(self, attr):
                    self.glView.removeItem(getattr(self, attr))
                    delattr(self, attr)
        self.glView.update()

    def updateClipPlane(self, value):
        try:
            clip_pos = (value / 100) * 30
            mask = self.scatter.pos[:, 2] <= clip_pos
            self.scatter.setData(pos=self.scatter.pos[mask],
                                 color=self.scatter.color[mask])
            self.logger.info(f"Clip plane updated to position {clip_pos}")
        except Exception as e:
            self.logger.error(f"Error updating clip plane: {str(e)}")


    def updateThreshold(self, value):
        # if value >= 1:
        #     self.logger.warning("Array compute error")
        #     return

        if self.data is not None:
            self.updateViews()
            self.create3DVisualization()
        else:
            self.logger.warning("No data available to update display threshold")

    def updateBlobThreshold(self, value):
       self.filter_blobs()


    def loadData(self, filename):
        try:
            self.data = self.data_manager.load_data(filename)
            self.updateUIForNewData()
            self.updateViews()
            self.create3DVisualization()
            self.autoScaleViews()
            self.updateRawDataViewer()  # Add this line
            self.logger.info(f"Data loaded from {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")


    def saveData(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "TIFF Files (*.tiff);;NumPy Files (*.npy)")
            if filename:
                self.data_manager.save_data(filename)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")

    def import_microscope_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Microscope Data", "",
                                                   "All Files (*);;TIFF Files (*.tif *.tiff)")
        if file_path:
            try:
                self.data, metadata = self.data_manager.import_microscope_data(file_path)
                self.updateUIForNewData()
                self.updateViews()
                self.create3DVisualization()
                self.updateRawDataViewer()  # Add this line
                self.display_metadata(metadata)
            except Exception as e:
                QMessageBox.critical(self, "Import Error", str(e))

    def toggleDownsamplingControls(self):
        enabled = self.downsamplingCheckBox.isChecked()
        self.downsamplingSpinBox.setEnabled(enabled)
        self.updateVisualization()

    def downsampleData(self, data, max_points):
        total_points = np.prod(data.shape[1:])  # Exclude channel dimension
        if total_points <= max_points:
            return data

        downsample_factor = int(np.ceil(np.cbrt(total_points / max_points)))
        downsampled = data[:, ::downsample_factor, ::downsample_factor, ::downsample_factor]
        return downsampled


    def togglePlayback(self):
        if not hasattr(self, 'playbackTimer'):
            self.playbackTimer = QTimer(self)
            self.playbackTimer.timeout.connect(self.advanceTimePoint)

        if self.playbackTimer.isActive():
            self.playbackTimer.stop()
            self.playPauseButton.setText("Play")
        else:
            self.playbackTimer.start(int(1000 / self.speedSpinBox.value()))
            self.playPauseButton.setText("Pause")

    def updatePlaybackSpeed(self, value):
        if hasattr(self, 'playbackTimer') and self.playbackTimer.isActive():
            self.playbackTimer.setInterval(int(1000 / value))


    def setChannelColor(self, channel, color):
        if channel < len(self.channel_colors):
            self.channel_colors[channel] = color
            self.updateViews()
            self.create3DVisualization()
            self.updateBlobVisualization()  # Add this line to update blob colors


    def getChannelColor(self, channel):
        if channel < len(self.channel_colors):
            color = self.channel_colors[channel]
            return (color.redF(), color.greenF(), color.blueF(), 1.0)
        return (1.0, 1.0, 1.0, 1.0)  # Default to white if channel is out of range




    def updateUIForNewData(self):
        if self.data is not None:
            self.timeSlider.setMaximum(self.data.shape[0] - 1)
            num_channels = self.data.shape[1]

            if self.current_simulation_type == 'shape':
                channel_names = ['Shapes', 'Empty 1', 'Empty 2', 'Empty 3']
            elif self.current_simulation_type == 'original':
                channel_names = ['Membrane', 'Nucleus', 'ER', 'Mitochondria', 'Actin', 'Microtubules', 'Calcium']
            else:  # 'enhanced'
                channel_names = ['Membrane', 'Nucleus', 'Cytoplasm', 'Extracellular', 'Proteins']

            self.ui_manager.update_channel_controls(num_channels, channel_names)

            self.updateViews()
            self.create3DVisualization()
        else:
            self.logger.warning("No data available to update UI")



    def closeEvent(self, event):
        # Stop the playback timer
        if hasattr(self, 'playbackTimer'):
            self.playbackTimer.stop()

        # Perform any other cleanup operations here
        # For example, you might want to save application settings

        # Log the application closure
        self.logger.info("Application closed")

        # Accept the event to allow the window to close
        event.accept()

        # Make sure to close the biological simulation window when the main window is closed
        if self.biological_simulation_window:
            self.biological_simulation_window.close()

        if hasattr(self, 'enhanced_biological_simulation_window'):
            self.enhanced_biological_simulation_window.close()

        # Call the base class implementation
        super().closeEvent(event)

    def safeClose(self):
        self.close()  # This will trigger the closeEvent

    def exportProcessedData(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Processed Data", "", "TIFF Files (*.tiff);;NumPy Files (*.npy)")
        if filename:
            if filename.endswith('.tiff'):
                tifffile.imwrite(filename, self.data)
            elif filename.endswith('.npy'):
                np.save(filename, self.data)

    def exportScreenshot(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png)")
        if filename:
            screen = QApplication.primaryScreen()
            screenshot = screen.grabWindow(self.winId())
            screenshot.save(filename, 'png')

    def exportVideo(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Video", "", "MP4 Files (*.mp4)")
        if filename:
            import imageio
            writer = imageio.get_writer(filename, fps=10)
            for t in range(self.data.shape[0]):
                self.timeSlider.setValue(t)
                QApplication.processEvents()
                screen = QApplication.primaryScreen()
                screenshot = screen.grabWindow(self.winId())
                writer.append_data(screenshot.toImage().convertToFormat(QImage.Format_RGB888).bits().asstring(screenshot.width() * screenshot.height() * 3))
            writer.close()

    def detect_blobs(self):
        if self.data is None:
            self.logger.warning("No data available for blob detection")
            return

        max_time =  self.timeSlider.maximum()+1
        num_channels = self.data.shape[1]

        # Get blob detection parameters from UI
        max_sigma = self.maxSigmaSpinBox.value()
        num_sigma = self.numSigmaSpinBox.value()
        #threshold = self.blobThresholdSpinBox.value()

        all_blobs = []
        for channel in range(num_channels):
            for t in range(max_time):
                # Get the current 3D volume for this channel
                volume = self.data[t, channel]

                # Detect blobs
                blobs = blob_log(volume, max_sigma=max_sigma, num_sigma=num_sigma)

                # Calculate intensity for each blob
                for blob in blobs:
                    y, x, z, r = blob
                    y, x, z = int(y), int(x), int(z)
                    r = int(r)

                    # Define a small region around the blob center
                    y_min, y_max = max(0, y-r), min(volume.shape[0], y+r+1)
                    x_min, x_max = max(0, x-r), min(volume.shape[1], x+r+1)
                    z_min, z_max = max(0, z-r), min(volume.shape[2], z+r+1)

                    # Extract the region
                    region = volume[y_min:y_max, x_min:x_max, z_min:z_max]

                    # Calculate the intensity (you can use different measures here)
                    intensity = np.mean(region)  # or np.max(region), np.sum(region), etc.

                    # Add blob information including intensity
                    all_blobs.append([y, x, z, r, channel, t, intensity])

        # Convert to numpy array
        all_blobs = np.array(all_blobs)

        # Store the original blob data
        self.original_blobs = all_blobs

        # Filter blobs based on threshold -> creates self.all_detected_blobs
        self.filter_blobs()

        self.logger.info(f"Detected {len(all_blobs)} blobs across all channels")

        # Store all detected blobs
        #self.all_detected_blobs = all_blobs

        # Display results
        self.display_blob_results(self.all_detected_blobs)

        # Visualize blobs
        self.updateBlobVisualization()

        # Show the blob results button and update its text
        self.showBlobResultsButton.setVisible(True)
        self.showBlobResultsButton.setText("Show Blob Results")

        return all_blobs


    def filter_blobs(self):
        if not hasattr(self, 'original_blobs'):
            return

        threshold = self.blobThresholdSpinBox.value()
        self.all_detected_blobs = self.original_blobs[self.original_blobs[:, 6] > threshold]

        # Update visualization
        self.updateBlobVisualization()


    def display_blob_results(self, blobs):
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("Blob Detection Results")
        layout = QVBoxLayout(result_dialog)

        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(["X", "Y", "Z", "Size", "Channel", "Time", "Intensity"])
        table.setRowCount(len(blobs))

        for i, blob in enumerate(blobs):
            y, x, z, r, channel, t, intensity = blob
            table.setItem(i, 0, QTableWidgetItem(f"{x:.2f}"))
            table.setItem(i, 1, QTableWidgetItem(f"{y:.2f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{z:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{r:.2f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{int(channel)}"))
            table.setItem(i, 5, QTableWidgetItem(f"{int(t)}"))
            table.setItem(i, 6, QTableWidgetItem(f"{intensity:.2f}"))

        layout.addWidget(table)

        close_button = QPushButton("Close")
        close_button.clicked.connect(result_dialog.close)
        layout.addWidget(close_button)

        result_dialog.exec_()

    def updateBlobVisualization(self):
        if not hasattr(self, 'all_detected_blobs') or self.all_detected_blobs is None:
            return

        current_time = self.timeSlider.value()

        if self.showAllBlobsCheck.isChecked():
            blobs_to_show = self.all_detected_blobs
        else:
            blobs_to_show = self.all_detected_blobs[self.all_detected_blobs[:, 5] == current_time]

        self.visualization_manager.visualize_blobs(blobs_to_show, current_time, self.channel_colors)

    def toggleBlobResults(self):
        if self.blob_results_dialog.isVisible():
            self.blob_results_dialog.hide()
            self.showBlobResultsButton.setText("Show Blob Results")
        else:
            if hasattr(self, 'all_detected_blobs'):
                self.blob_results_dialog.update_results(self.all_detected_blobs)
            self.blob_results_dialog.show()
            self.showBlobResultsButton.setText("Hide Blob Results")

    def clearDetectedBlobs(self):
        if hasattr(self, 'all_detected_blobs'):
            del self.all_detected_blobs
        self.updateBlobVisualization()
        self.blob_results_dialog.hide()
        self.showBlobResultsButton.setVisible(False)

    def analyzeBlobsasdkjfb(self):
        if hasattr(self, 'all_detected_blobs') and self.all_detected_blobs is not None:
            blob_analyzer = BlobAnalyzer(self.all_detected_blobs)
            analysis_dialog = BlobAnalysisDialog(blob_analyzer, self)
            analysis_dialog.setWindowTitle("Blob Analysis Results")
            analysis_dialog.setGeometry(100, 100, 800, 600)
            analysis_dialog.show()
        else:
            QMessageBox.warning(self, "No Blobs Detected", "Please detect blobs before running analysis.")

    def showTimeSeriesAnalysis(self):
        if hasattr(self, 'all_detected_blobs') and self.all_detected_blobs is not None:
            dialog = TimeSeriesDialog(BlobAnalyzer(self.all_detected_blobs), self)
            dialog.exec_()
        else:
            QMessageBox.warning(self, "No Data", "Please detect blobs first.")

    def autoScaleViews(self):
        if self.data is None:
            return

        # Get the bounds of the data
        z, y, x = self.data.shape[2:]  # Assuming shape is (t, c, z, y, x)
        center = QVector3D(x/2, y/2, z/2)

        # Calculate the diagonal of the bounding box
        diagonal = np.sqrt(x**2 + y**2 + z**2)

        # Set the camera position for both views
        for view in [self.glView, self.blobGLView]:
            view.setCameraPosition(pos=center, distance=diagonal*1.2, elevation=30, azimuth=45)
            view.opts['center'] = center

        self.glView.update()
        self.blobGLView.update()



    def connectViewEvents(self):
        for view in [self.glView, self.blobGLView]:
            if view is not None:
                view.installEventFilter(self)
        self.logger.debug("View events connected")

    def eventFilter(self, source, event):
        if source in [self.glView, self.blobGLView]:
            if event.type() == QEvent.MouseButtonPress:
                self.on3DViewMousePress(event, source)
                return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.on3DViewMouseRelease(event, source)
                return True
            elif event.type() == QEvent.MouseMove:
                self.on3DViewMouseMove(event, source)
                return True
            elif event.type() == QEvent.Wheel:
                self.on3DViewWheel(event, source)
                return True
        return super().eventFilter(source, event)

    @pyqtSlot(QEvent)
    def on3DViewMousePress(self, event, source):
        self.lastPos = event.pos()
        self.logger.debug(f"Mouse press event on {source}")

    @pyqtSlot(QEvent)
    def on3DViewMouseRelease(self, event, source):
        self.lastPos = None
        self.logger.debug(f"Mouse release event on {source}")

    @pyqtSlot(QEvent)
    def on3DViewMouseMove(self, event, source):
        if self.lastPos is None:
            return

        diff = event.pos() - self.lastPos
        self.lastPos = event.pos()

        self.logger.debug(f"Mouse move event on {source}")

        if event.buttons() == Qt.LeftButton:
            self.rotate3DViews(diff.x(), diff.y(), source)
        elif event.buttons() == Qt.MidButton:
            self.pan3DViews(diff.x(), diff.y(), source)

    @pyqtSlot(QEvent)
    def on3DViewWheel(self, event, source):
        delta = event.angleDelta().y()
        self.logger.debug(f"Wheel event on {source}")
        self.zoom3DViews(delta, source)

    def rotate3DViews(self, dx, dy, active_view):
        views_to_update = [active_view]
        if self.syncViewsCheck.isChecked():
            views_to_update = [self.glView, self.blobGLView]

        for view in views_to_update:
            if view is not None and hasattr(view, 'opts'):
                view.opts['elevation'] -= dy * 0.5
                view.opts['azimuth'] += dx * 0.5
                view.update()
            else:
                self.logger.error(f"Invalid view object: {view}")

    def pan3DViews(self, dx, dy, active_view):
        views_to_update = [active_view]
        if self.syncViewsCheck.isChecked():
            views_to_update = [self.glView, self.blobGLView]

        for view in views_to_update:
            if view is not None and hasattr(view, 'pan'):
                view.pan(dx, dy, 0, relative='view')
            else:
                self.logger.error(f"Invalid view object for panning: {view}")

    def zoom3DViews(self, delta, active_view):
        views_to_update = [active_view]
        if self.syncViewsCheck.isChecked():
            views_to_update = [self.glView, self.blobGLView]

        for view in views_to_update:
            if view is not None and hasattr(view, 'opts'):
                view.opts['fov'] *= 0.999**delta
                view.update()
            else:
                self.logger.error(f"Invalid view object for zooming: {view}")

    def check_view_state(self):
        self.logger.debug(f"glView state: {self.glView}, has opts: {hasattr(self.glView, 'opts')}")
        self.logger.debug(f"blobGLView state: {self.blobGLView}, has opts: {hasattr(self.blobGLView, 'opts')}")
        self.logger.debug(f"Sync checked: {self.syncViewsCheck.isChecked()}")


    def setTopDownView(self):
        for view in [self.glView, self.blobGLView]:
            view.setCameraPosition(elevation=90, azimuth=0)
            view.update()

    def setSideView(self):
        for view in [self.glView, self.blobGLView]:
            view.setCameraPosition(elevation=0, azimuth=0)
            view.update()

    def setFrontView(self):
        for view in [self.glView, self.blobGLView]:
            view.setCameraPosition(elevation=0, azimuth=90)
            view.update()



    def display_metadata(self, metadata):
        if metadata is None:
            QMessageBox.warning(self, "Metadata", "No metadata available for this file.")
            return

        # Create a new dialog to display metadata
        dialog = QDialog(self)
        dialog.setWindowTitle("Metadata")
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        # Format metadata for display
        metadata_text = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
        text_edit.setText(metadata_text)

        layout.addWidget(text_edit)
        dialog.setLayout(layout)
        dialog.exec_()

    def runEnhancedBiologicalSimulation(self, params):
        try:
            self.data = None  # Reset data at the start of simulation
            self.current_simulation_type = 'enhanced'

            # Create CellularEnvironment
            env_size = params['environment_size']
            simulator = EnhancedBiologicalSimulator(env_size, num_time_points=100)

            # Create and add cells
            for _ in range(params['num_cells']):
                cell_size = (20, 20, 20)  # You might want to make this configurable
                position = tuple(np.random.randint(0, s - cs) for s, cs in zip(env_size, cell_size))
                simulator.add_cell(position, cell_size)

            # Add a global protein if diffusion is enabled
            if params['protein_diffusion']['enabled']:
                initial_concentration = np.zeros(env_size)
                initial_concentration[tuple(s // 2 for s in env_size)] = 1.0
                simulator.add_global_protein('diffusing_protein', initial_concentration, params['protein_diffusion']['coefficient'])

            # Run simulation
            for state in simulator.run_simulation():
                self.processSimulationState(state)

            # Add debugging output
            if self.data is not None:
                self.logger.info(f"Final data shape: {self.data.shape}")
                for i in range(self.data.shape[1]):
                    channel_data = self.data[:, i]
                    self.logger.info(f"Channel {i} - min: {np.min(channel_data)}, max: {np.max(channel_data)}, mean: {np.mean(channel_data)}")

            self.updateUIForNewData()
            self.updateViews()
            self.create3DVisualization()

        except Exception as e:
            self.handleSimulationError(e)


    def processSimulationState(self, state):
        self.logger.info(f"Processing state with keys: {state.keys()}")
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                self.logger.info(f"State['{key}']: shape={value.shape}, dtype={value.dtype}, min={np.min(value)}, max={np.max(value)}, mean={np.mean(value)}")
            elif isinstance(value, dict):
                self.logger.info(f"State['{key}']: dict with keys {value.keys()}")
            else:
                self.logger.info(f"State['{key}']: type={type(value)}")

        # Keep each component separate
        membrane = state['membrane'].astype(float)
        nucleus = state['nucleus'].astype(float)
        cytoplasm = state['cytoplasm'].astype(float)
        extracellular = state['extracellular_space'].astype(float)

        # Combine proteins into one channel (if present)
        proteins = np.zeros_like(membrane)
        if 'proteins' in state:
            for protein_name, concentration in state['proteins'].items():
                proteins += concentration

        # Combine all channels
        combined_state = np.stack([membrane, nucleus, cytoplasm, extracellular, proteins], axis=0)

        # Add the combined state to our data
        if self.data is None:
            self.data = combined_state[np.newaxis, ...]
        else:
            self.data = np.concatenate([self.data, combined_state[np.newaxis, ...]], axis=0)

        self.logger.info(f"Processed state shape: {combined_state.shape}")
        for i, name in enumerate(['membrane', 'nucleus', 'cytoplasm', 'extracellular', 'proteins']):
            self.logger.info(f"Channel {i} ({name}) - min: {np.min(combined_state[i])}, max: {np.max(combined_state[i])}, mean: {np.mean(combined_state[i])}")


    def runShapeSimulation(self, params):
        try:
            self.data = None  # Reset data at the start of simulation
            self.current_simulation_type = 'shape'

            simulator = ShapeSimulator((100, 100, 100))  # Make sure this matches your desired simulation size

            # Add shapes based on params
            for shape_params in params['shapes']:
                if shape_params['type'] == 'sphere':
                    shape = Sphere(shape_params['position'], shape_params['size'], shape_params['color'])
                elif shape_params['type'] == 'cube':
                    shape = Cube(shape_params['position'], shape_params['size'], shape_params['color'])
                else:
                    raise ValueError(f"Unknown shape type: {shape_params['type']}")

                if shape_params['movement'] == 'random':
                    movement = RandomWalk(shape_params['speed'])
                elif shape_params['movement'] == 'linear':
                    movement = LinearMotion(shape_params.get('velocity', [1, 0, 0]))
                else:
                    raise ValueError(f"Unknown movement type: {shape_params['movement']}")

                simulator.add_shape(shape, movement)

            # Add interactions
            for interaction_params in params['interactions']:
                if interaction_params['type'] == 'attraction':
                    interaction = Attraction(interaction_params.get('strength', 1.0))
                elif interaction_params['type'] == 'repulsion':
                    interaction = Repulsion(interaction_params.get('strength', 1.0), interaction_params.get('range', 10))
                elif interaction_params['type'] == 'none':
                    continue  # No interaction to add
                else:
                    raise ValueError(f"Unknown interaction type: {interaction_params['type']}")
                simulator.add_interaction(interaction)

            # Run simulation
            num_steps = params['num_steps']
            dt = params['dt']
            resolution = params.get('resolution', 10)  # Get resolution from params or use default
            for _ in range(num_steps):
                simulator.update(dt)
                state = simulator.get_state(resolution)
                self.processShapeSimulationState(state)

            self.updateUIForNewData()
            self.updateViews()
            self.create3DVisualization()

        except Exception as e:
            self.handleSimulationError(e)

    def processShapeSimulationState(self, state):
        # Convert boolean state to float
        float_state = state.astype(float)

        # Create additional dummy channels to match the expected format
        dummy_channels = np.zeros_like(float_state)

        # Stack the original state with dummy channels
        combined_state = np.stack([float_state, dummy_channels, dummy_channels, dummy_channels], axis=0)

        if self.data is None:
            self.data = combined_state[np.newaxis, ...]
        else:
            self.data = np.concatenate([self.data, combined_state[np.newaxis, ...]], axis=0)

        self.logger.info(f"Processed state shape: {combined_state.shape}")

    def toggle_intensity_profile_tool(self, checked):
        if checked:
            self.visualization_manager.enable_intensity_profile_tool(self.imageViewXY)
        else:
            if self.visualization_manager.profile_roi is not None:
                self.imageViewXY.removeItem(self.visualization_manager.profile_roi)
                self.visualization_manager.profile_roi = None
            if self.visualization_manager.profile_window is not None:
                self.visualization_manager.profile_window.close()
                self.visualization_manager.profile_window = None

    def createRawDataMenu(self):
        rawDataMenu = self.menuBar().addMenu('&Raw Data')

        showRawDataAction = rawDataMenu.addAction('Show Raw Data Viewer')
        showRawDataAction.triggered.connect(self.showRawDataViewer)

        setPropertiesAction = rawDataMenu.addAction('Set Data Properties')
        setPropertiesAction.triggered.connect(self.setDataProperties)

        processingMenu = rawDataMenu.addMenu('Image Processing')

        gaussianBlurAction = processingMenu.addAction('Gaussian Blur')
        gaussianBlurAction.triggered.connect(self.applyGaussianBlur)

        medianFilterAction = processingMenu.addAction('Median Filter')
        medianFilterAction.triggered.connect(self.applyMedianFilter)

    def setDataProperties(self):
        dialog = DataPropertiesDialog(self)
        if dialog.exec_():
            self.data_properties = dialog.get_properties()
            if self.raw_data is not None:
                self.updateRawDataViewer()

    def showRawDataViewer(self):
        if self.raw_data_viewer is None:
            self.updateRawDataViewer()
        else:
            self.raw_data_viewer.show()

    def updateRawDataViewer(self):
        if self.raw_data_viewer is None:
            self.raw_data_viewer = RawDataViewer()

        if self.data is not None:
            # self.data is assumed to be in the format (t, c, z, y, x)
            properties = {
                'num_z_slices': self.data.shape[2],
                'num_channels': self.data.shape[1],
                'pixel_size_xy': 1.0,  # You might want to make this configurable
                'pixel_size_z': 1.0,   # You might want to make this configurable
                'slice_angle': 90      # You might want to make this configurable
            }
            self.raw_data_viewer.setData(self.data, properties)

        self.raw_data_viewer.show()


    def applyGaussianBlur(self):
        if self.data is not None:
            self.data = self.image_processor.gaussian_blur(self.data)
            self.updateViews()
            if self.raw_data_viewer:
                self.raw_data_viewer.setData(self.data)

    def applyMedianFilter(self):
        if self.data is not None:
            self.data = self.image_processor.median_filter(self.data)
            self.updateViews()
            if self.raw_data_viewer:
                self.raw_data_viewer.setData(self.data)


##############################################################################

def main():
    app = QApplication(sys.argv)
    viewer = LightsheetViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
