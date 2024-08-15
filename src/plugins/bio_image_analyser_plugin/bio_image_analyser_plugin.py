#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:54:57 2024

@author: george
"""
import os
import sys

# Add the directory containing this plugin to the Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

# plugins/bio_image_analyzer/bio_image_analyzer.py

import numpy as np
import pandas as pd
from skimage import filters, segmentation, measure, feature, exposure, morphology
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QSpinBox, QDoubleSpinBox, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from plugin_base import Plugin

class BioImageAnalyzer(Plugin):
    def __init__(self, microview):
        self.microview = microview
        self.name = "BioImageAnalyzer"
        self.widget = None

    def run(self):
        if self.widget is None:
            self.widget = BioImageAnalyzerWidget(self.microview)
        self.widget.show()

class BioImageAnalyzerWidget(QWidget):
    def __init__(self, microview):
        super().__init__()
        self.microview = microview
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Cell Segmentation",
            "Nucleus Detection",
            "Spot Detection",
            "Fiber Tracing",
            "Colocalization Analysis",
            "Intensity Measurement",
            "Morphological Analysis"
        ])
        layout.addWidget(QLabel("Analysis Method:"))
        layout.addWidget(self.method_combo)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        layout.addWidget(QLabel("Threshold:"))
        layout.addWidget(self.threshold_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 1000)
        self.min_size_spin.setValue(50)
        layout.addWidget(QLabel("Minimum Size:"))
        layout.addWidget(self.min_size_spin)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)
        layout.addWidget(self.analyze_button)

        self.result_figure = Figure(figsize=(5, 4), dpi=100)
        self.result_canvas = FigureCanvas(self.result_figure)
        layout.addWidget(self.result_canvas)

        self.setLayout(layout)
        self.setWindowTitle("BioImageAnalyzer")

    def analyze(self):
        if self.microview.window_management.current_window is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        image = self.microview.window_management.current_window.image
        method = self.method_combo.currentText()
        threshold = self.threshold_spin.value()
        min_size = self.min_size_spin.value()

        if image.ndim == 3:  # Time series
            result = self.analyze_time_series(image, method, threshold, min_size)
        else:  # Single frame
            result = self.analyze_single_frame(image, method, threshold, min_size)

        self.display_result(result, method)

    def analyze_single_frame(self, image, method, threshold, min_size):
        if method == "Cell Segmentation":
            return self.cell_segmentation(image, threshold, min_size)
        elif method == "Nucleus Detection":
            return self.nucleus_detection(image, threshold, min_size)
        elif method == "Spot Detection":
            return self.spot_detection(image, threshold, min_size)
        elif method == "Fiber Tracing":
            return self.fiber_tracing(image, threshold)
        elif method == "Colocalization Analysis":
            return self.colocalization_analysis(image)
        elif method == "Intensity Measurement":
            return self.intensity_measurement(image)
        elif method == "Morphological Analysis":
            return self.morphological_analysis(image, threshold, min_size)

    def analyze_time_series(self, image_stack, method, threshold, min_size):
        results = []
        for i in range(image_stack.shape[0]):
            result = self.analyze_single_frame(image_stack[i], method, threshold, min_size)
            results.append(result)
        return results

    def cell_segmentation(self, image, threshold, min_size):
        edges = filters.sobel(image)
        markers = filters.threshold_otsu(edges)
        segmentation_mask = segmentation.watershed(edges, markers, mask=image)
        labeled_cells = measure.label(segmentation_mask)
        properties = measure.regionprops(labeled_cells, image)
        return labeled_cells, properties

    def nucleus_detection(self, image, threshold, min_size):
        binary = image > filters.threshold_otsu(image)
        labeled_nuclei = measure.label(binary)
        properties = measure.regionprops(labeled_nuclei, image)
        return labeled_nuclei, properties

    def spot_detection(self, image, threshold, min_size):
        local_max = feature.peak_local_max(image, min_distance=min_size, threshold_abs=threshold)
        spots = np.zeros(image.shape, dtype=bool)
        spots[tuple(local_max.T)] = True
        labeled_spots = measure.label(spots)
        properties = measure.regionprops(labeled_spots, image)
        return labeled_spots, properties

    def fiber_tracing(self, image, threshold):
        binary = image > filters.threshold_otsu(image)
        skeleton = morphology.skeletonize(binary)
        return skeleton

    def colocalization_analysis(self, image):
        if image.shape[-1] != 2:
            raise ValueError("Colocalization analysis requires a 2-channel image")
        channel1 = image[..., 0]
        channel2 = image[..., 1]
        pearson_corr = np.corrcoef(channel1.flatten(), channel2.flatten())[0, 1]
        manders_m1 = np.sum(channel1[channel2 > 0]) / np.sum(channel1)
        manders_m2 = np.sum(channel2[channel1 > 0]) / np.sum(channel2)
        return pearson_corr, manders_m1, manders_m2

    def intensity_measurement(self, image):
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        min_intensity = np.min(image)
        max_intensity = np.max(image)
        return mean_intensity, std_intensity, min_intensity, max_intensity

    def morphological_analysis(self, image, threshold, min_size):
        binary = image > filters.threshold_otsu(image)
        labeled_objects = measure.label(binary)
        properties = measure.regionprops(labeled_objects, image)
        return labeled_objects, properties

    def display_result(self, result, method):
        self.result_figure.clear()
        ax = self.result_figure.add_subplot(111)

        if method in ["Cell Segmentation", "Nucleus Detection", "Spot Detection", "Morphological Analysis"]:
            if isinstance(result[0], np.ndarray):  # Single frame
                ax.imshow(result[0], cmap='viridis')
                ax.set_title(f"{method} Result")
            else:  # Time series
                ax.plot(range(len(result)), [len(r[1]) for r in result])
                ax.set_title(f"{method}: Object Count over Time")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Object Count")

        elif method == "Fiber Tracing":
            if isinstance(result, np.ndarray):  # Single frame
                ax.imshow(result, cmap='gray')
                ax.set_title("Fiber Tracing Result")
            else:  # Time series
                ax.plot(range(len(result)), [np.sum(r) for r in result])
                ax.set_title("Fiber Length over Time")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Total Fiber Length")

        elif method == "Colocalization Analysis":
            if len(result) == 3:  # Single frame
                pearson, m1, m2 = result
                ax.bar(['Pearson', "Manders' M1", "Manders' M2"], [pearson, m1, m2])
                ax.set_title("Colocalization Analysis")
            else:  # Time series
                pearson_values = [r[0] for r in result]
                ax.plot(range(len(result)), pearson_values)
                ax.set_title("Pearson Correlation over Time")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Pearson Correlation")

        elif method == "Intensity Measurement":
            if len(result) == 4:  # Single frame
                mean, std, min_val, max_val = result
                ax.bar(['Mean', 'Std Dev', 'Min', 'Max'], [mean, std, min_val, max_val])
                ax.set_title("Intensity Measurements")
            else:  # Time series
                mean_values = [r[0] for r in result]
                ax.plot(range(len(result)), mean_values)
                ax.set_title("Mean Intensity over Time")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Mean Intensity")

        self.result_canvas.draw()

        # Update the current image in MicroView
        if isinstance(result[0], np.ndarray):
            self.microview.loadImage(result[0])



# This line is crucial for the plugin loader to work
Plugin = BioImageAnalyzer
