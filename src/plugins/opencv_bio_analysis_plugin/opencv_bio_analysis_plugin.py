#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:25:39 2024

@author: george
"""

# opencv_bio_analysis_plugin.py

import os
import sys

# Add the directory containing this plugin to the Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QCheckBox,
                             QTextEdit, QSplitter, QHBoxLayout)
from PyQt5.QtCore import Qt
from skimage import filters, measure, exposure
from scipy import ndimage
from plugin_base import Plugin
import pyqtgraph as pg

class OpenCVBioAnalysisPlugin(Plugin):
    def __init__(self, microview):
        super().__init__(microview)
        self.name = "OpenCV Bio Analysis"
        self.widget = None
        self.logger = self.setup_plugin_logger()
        self.results_text = None
        self.is_timeseries = False

    def run(self):
        self.widget = QWidget()
        layout = QVBoxLayout()

        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems([
            "Fluorescence Intensity",
            "Background Subtraction",
            "Spot Detection",
            "Temporal Projection",
            "Bleach Correction"
        ])
        layout.addWidget(QLabel("Select Analysis:"))
        layout.addWidget(self.analysis_combo)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.1)
        layout.addWidget(QLabel("Threshold (relative):"))
        layout.addWidget(self.threshold_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 1000)
        self.min_size_spin.setValue(5)
        layout.addWidget(QLabel("Min Spot Size:"))
        layout.addWidget(self.min_size_spin)

        self.process_timeseries = QCheckBox("Process entire time series")
        layout.addWidget(self.process_timeseries)

        run_button = QPushButton("Run Analysis")
        run_button.clicked.connect(self.run_analysis)
        layout.addWidget(run_button)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        self.widget.setLayout(layout)
        self.widget.show()

    def get_current_image(self):
        try:
            if hasattr(self.microview.window_manager.current_window, 'flika_window'):
                image = self.microview.window_manager.current_window.flika_window.image
            else:
                image = self.microview.window_manager.current_window.image

            self.logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")

            # Check if it's a time series
            if image.ndim == 3 and image.shape[0] > 1:
                self.is_timeseries = True
                if not self.process_timeseries.isChecked():
                    # If not processing the whole series, just take the first frame
                    image = image[0]
                    self.is_timeseries = False
            else:
                self.is_timeseries = False

            return image

        except Exception as e:
            self.logger.error(f"Error getting current image: {str(e)}")
            return None

    def run_analysis(self):
        image = self.get_current_image()
        if image is None:
            QMessageBox.warning(self.widget, "Error", "No image loaded")
            return

        analysis = self.analysis_combo.currentText()
        threshold = self.threshold_spin.value()
        min_size = self.min_size_spin.value()

        try:
            if analysis == "Fluorescence Intensity":
                result, text_result = self.fluorescence_intensity(image)
            elif analysis == "Background Subtraction":
                result, text_result = self.background_subtraction(image)
            elif analysis == "Spot Detection":
                result, text_result = self.spot_detection(image, threshold, min_size)
            elif analysis == "Temporal Projection":
                result, text_result = self.temporal_projection(image)
            elif analysis == "Bleach Correction":
                result, text_result = self.bleach_correction(image)
            else:
                raise ValueError(f"Unknown analysis type: {analysis}")

            self.results_text.setText(text_result)
            self.show_result(result, f"{analysis} Result")
            self.logger.info(f"Completed {analysis}")
        except Exception as e:
            self.logger.error(f"Error in {analysis}: {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"Analysis failed: {str(e)}")

    def fluorescence_intensity(self, image):
        if not self.is_timeseries:
            mean_intensity = np.mean(image)
            text_result = f"Mean Intensity: {mean_intensity:.4f}"
        else:
            mean_intensities = np.mean(image, axis=(1,2))
            text_result = "Mean Intensities over time:\n" + \
                          "\n".join([f"Frame {i}: {intensity:.4f}" for i, intensity in enumerate(mean_intensities)])
        return image, text_result

    def background_subtraction(self, image):
        def subtract_background(frame):
            background = filters.gaussian(frame, sigma=50)
            return frame - background

        if not self.is_timeseries:
            result = subtract_background(image)
        else:
            result = np.array([subtract_background(frame) for frame in image])

        text_result = "Background subtraction completed."
        return result, text_result

    def spot_detection(self, image, threshold, min_size):
        def detect_spots(frame):
            thresh = filters.threshold_local(frame, block_size=51)
            binary = frame > (thresh + threshold)
            labeled = measure.label(binary)
            properties = measure.regionprops(labeled, frame)
            spots = [region for region in properties if region.area >= min_size]

            result = exposure.rescale_intensity(frame, out_range=(0, 1))
            result = np.dstack([result, result, result])
            for spot in spots:
                y, x = spot.centroid
                cv2.circle(result, (int(x), int(y)), 3, (1, 0, 0), -1)

            return spots, result

        if not self.is_timeseries:
            spots, result = detect_spots(image)
            text_result = f"Detected {len(spots)} spots."
        else:
            all_spots = []
            result = []
            for i in range(image.shape[0]):
                frame_spots, frame_result = detect_spots(image[i])
                all_spots.append(frame_spots)
                result.append(frame_result)
            result = np.array(result)
            text_result = "Spot detection completed for all frames.\n" + \
                          "\n".join([f"Frame {i}: {len(spots)} spots" for i, spots in enumerate(all_spots)])

        self.logger.info(f"Spot detection result shape: {result.shape}")
        return result, text_result

    def temporal_projection(self, image):
        if not self.is_timeseries:
            raise ValueError("Temporal projection requires a time series")

        max_proj = np.max(image, axis=0)
        mean_proj = np.mean(image, axis=0)
        std_proj = np.std(image, axis=0)

        result = np.dstack([max_proj, mean_proj, std_proj])

        text_result = "Temporal projection completed.\n" \
                      f"Max projection mean: {np.mean(max_proj):.4f}\n" \
                      f"Mean projection mean: {np.mean(mean_proj):.4f}\n" \
                      f"Std projection mean: {np.mean(std_proj):.4f}"

        return result, text_result

    def bleach_correction(self, image):
        if not self.is_timeseries:
            raise ValueError("Bleach correction requires a time series")

        mean_intensities = np.mean(image, axis=(1,2))
        correction_factors = np.max(mean_intensities) / mean_intensities
        result = image * correction_factors[:, np.newaxis, np.newaxis]

        text_result = "Bleach correction completed.\n" \
                      f"Initial mean intensity: {mean_intensities[0]:.4f}\n" \
                      f"Final mean intensity: {mean_intensities[-1]:.4f}\n" \
                      f"Correction factor range: {np.min(correction_factors):.4f} - {np.max(correction_factors):.4f}"

        return result, text_result

    def show_result(self, result, title):
        try:
            self.logger.info(f"Showing result with shape: {result.shape}")

            if isinstance(result, np.ndarray):
                if result.ndim == 2:
                    # For 2D results (images)
                    pg_image = pg.image(result, title=title)
                    self.microview.window_manager.add_window(pg_image.window())
                elif result.ndim == 3:
                    if result.shape[2] == 3:
                        # For RGB results
                        pg_image = pg.image(np.transpose(result, (1, 0, 2)), title=title)
                        self.microview.window_manager.add_window(pg_image.window())
                    else:
                        # For time series results
                        image_view = pg.ImageView()
                        image_view.setImage(result)
                        image_view.setWindowTitle(title)
                        self.microview.window_manager.add_window(image_view)
                elif result.ndim == 4:
                    # For time series of RGB images
                    image_view = pg.ImageView()
                    image_view.setImage(result, xvals=np.arange(result.shape[0]))
                    image_view.setWindowTitle(title)
                    self.microview.window_manager.add_window(image_view)
                else:
                    raise ValueError(f"Unsupported result shape: {result.shape}")
            else:
                # For non-image results (e.g., graphs)
                plot = pg.plot(result, title=title)
                self.microview.window_manager.add_window(plot)

            self.logger.info(f"Successfully displayed result: {title}")
        except Exception as e:
            self.logger.error(f"Error displaying result: {str(e)}")
            QMessageBox.warning(self.widget, "Display Error", f"Unable to display result: {str(e)}")

# Add this to your plugin loading mechanism in MicroView
# plugins['OpenCV Bio Analysis'] = OpenCVBioAnalysisPlugin(microview_instance)

Plugin = OpenCVBioAnalysisPlugin
