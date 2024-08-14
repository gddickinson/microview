#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:35:30 2024

@author: george
"""

# scikit_analysis_console.py

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                             QSlider, QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QProgressDialog)
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
from scikit_image_analysis import ScikitImageAnalysis
import cv2  # Make sure you have this for the blob detection visualization
from PyQt5.QtCore import Qt, QTimer


class ScikitAnalysisConsole(QWidget):
    analysisCompleted = pyqtSignal(object)

    def __init__(self, image):
        super().__init__()
        self.image = image
        self.analysis = ScikitImageAnalysis()
        self.analysis.set_image(image)
        self.preview_image = self.image[0] if self.analysis.is_time_series else self.image
        if self.preview_image.ndim > 2:
            self.preview_image = self.preview_image[0]  # Take the first channel if it's multichannel
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        # Left side: controls
        control_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            'Gaussian Filter', 'Median Filter', 'Bilateral Filter', 'Sobel Edge',
            'Canny Edge', 'Otsu Threshold', 'Adaptive Threshold', 'Erode', 'Dilate',
            'Open', 'Close', 'Watershed Segmentation', 'Detect Blobs', 'Adjust Gamma',
            'Equalize Histogram', 'Resize', 'Rotate',
            'Unsharp Mask', 'Gaussian Gradient Magnitude', 'Laplacian',
            'Frangi Filter', 'Top-hat', 'Bottom-hat', 'Local Binary Pattern',
            'Contrast Stretch', 'Denoise (Non-local Means)', 'Hessian Matrix Eigenvalues'
        ])

        self.method_combo.currentIndexChanged.connect(self.update_controls)
        control_layout.addWidget(self.method_combo)

        self.param_layout = QVBoxLayout()
        control_layout.addLayout(self.param_layout)

        self.apply_button = QPushButton('Apply to All Frames')
        self.apply_button.clicked.connect(self.apply_to_all)
        control_layout.addWidget(self.apply_button)

        layout.addLayout(control_layout)

        # Right side: preview
        self.preview_plot = pg.ImageView()
        self.preview_plot.setImage(self.preview_image)
        layout.addWidget(self.preview_plot)

        self.setLayout(layout)
        self.setWindowTitle('Scikit-image Analysis Console')
        self.update_controls()

    def update_controls(self):
        # Remove all existing widgets from the layout
        for i in reversed(range(self.param_layout.count())):
            widget = self.param_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        method = self.method_combo.currentText()

        if method in ['Gaussian Filter', 'Canny Edge']:
            self.add_slider('Sigma', 0.1, 10, 1, 0.1)
        elif method in ['Median Filter', 'Erode', 'Dilate', 'Open', 'Close']:
            self.add_spinner('Size', 1, 20, 3)
        elif method == 'Bilateral Filter':
            self.add_slider('Sigma Color', 0.01, 1, 0.1, 0.01)
            self.add_slider('Sigma Spatial', 0.1, 10, 1, 0.1)
        elif method == 'Adaptive Threshold':
            self.add_spinner('Block Size', 3, 99, 35, 2)
            self.add_slider('Offset', -1, 1, 0, 0.01)
        elif method == 'Watershed Segmentation':
            self.add_spinner('Markers', 1, 100, 10)
        elif method == 'Detect Blobs':
            self.add_slider('Min Sigma', 0.1, 10, 1, 0.1)
            self.add_slider('Max Sigma', 1, 50, 30, 1)
            self.add_spinner('Num Sigma', 1, 20, 10)
            self.add_slider('Threshold', 0, 1, 0.1, 0.01)
        elif method == 'Adjust Gamma':
            self.add_slider('Gamma', 0.1, 5, 1, 0.1)
        elif method == 'Resize':
            self.add_spinner('Width', 1, 1000, self.preview_image.shape[1])
            self.add_spinner('Height', 1, 1000, self.preview_image.shape[0])
        elif method == 'Rotate':
            self.add_slider('Angle', -180, 180, 0, 1)
        elif method == 'Unsharp Mask':
            self.add_slider('Radius', 0.1, 5, 1, 0.1)
            self.add_slider('Amount', 0.1, 2, 1, 0.1)
        elif method == 'Gaussian Gradient Magnitude':
            self.add_slider('Sigma', 0.1, 5, 1, 0.1)
        elif method == 'Frangi Filter':
            self.add_slider('Beta1', 0.1, 1, 0.5, 0.1)
            self.add_slider('Beta2', 1, 30, 15, 1)
        elif method in ['Top-hat', 'Bottom-hat']:
            self.add_spinner('Size', 1, 20, 5)
        elif method == 'Local Binary Pattern':
            self.add_spinner('P', 4, 24, 8)
            self.add_slider('R', 0.1, 5, 1, 0.1)
        elif method == 'Contrast Stretch':
            self.add_slider('In Min', 0, 100, 0, 1)
            self.add_slider('In Max', 0, 100, 100, 1)
        elif method == 'Denoise (Non-local Means)':
            self.add_spinner('Patch Size', 1, 10, 5)
            self.add_spinner('Patch Distance', 1, 10, 6)
            self.add_slider('H', 0.01, 1, 0.1, 0.01)
        elif method == 'Hessian Matrix Eigenvalues':
            self.add_slider('Sigma', 0.1, 5, 1, 0.1)

        # Methods that don't need additional controls
        elif method in ['Sobel Edge', 'Otsu Threshold', 'Equalize Histogram']:
            pass
        else:
            print(f"Unknown method: {method}")

        self.update_preview()

    def add_slider(self, name, min_val, max_val, default, step):
        label = QLabel(name)
        self.param_layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(int((default - min_val) / (max_val - min_val) * 100))
        slider.valueChanged.connect(self.update_preview)
        slider.setProperty("min_val", min_val)
        slider.setProperty("max_val", max_val)
        self.param_layout.addWidget(slider)

        value_label = QLabel(f"{default:.2f}")
        self.param_layout.addWidget(value_label)

        def update_label(value):
            actual_value = min_val + (max_val - min_val) * value / 100
            value_label.setText(f"{actual_value:.2f}")

        slider.valueChanged.connect(update_label)

    def add_spinner(self, name, min_val, max_val, default, step=1):
        label = QLabel(name)
        self.param_layout.addWidget(label)

        if step == 1:
            spinner = QSpinBox()
        else:
            spinner = QDoubleSpinBox()
            spinner.setSingleStep(step)
        spinner.setMinimum(min_val)
        spinner.setMaximum(max_val)
        spinner.setValue(default)
        spinner.valueChanged.connect(self.update_preview)
        self.param_layout.addWidget(spinner)

    def get_param_values(self):
        values = {}
        i = 0
        while i < self.param_layout.count():
            label_widget = self.param_layout.itemAt(i).widget()
            if isinstance(label_widget, QLabel):
                label = label_widget.text()
                control = self.param_layout.itemAt(i+1).widget()
                if isinstance(control, QSlider):
                    value_label = self.param_layout.itemAt(i+2).widget()
                    value = float(value_label.text())
                    i += 3
                elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                    value = control.value()
                    i += 2
                else:
                    i += 1
                    continue
                values[label] = value
            else:
                i += 1
        return values

    def update_preview(self):
        method = self.method_combo.currentText()
        params = self.get_param_values()

        if method == 'Detect Blobs':
            result = self.analysis.detect_blobs(params['Min Sigma'], params['Max Sigma'],
                                                int(params['Num Sigma']), params['Threshold'])
            result_image = self.preview_image.copy()

            if isinstance(result, np.ndarray) and result.ndim == 2:  # Single frame result
                blobs = result
            elif isinstance(result, np.ndarray) and result.ndim == 3:  # Multiple frame result
                blobs = result[0]  # Take blobs from the first frame for preview
            else:
                print(f"Unexpected blob detection result type: {type(result)}, shape: {result.shape}")
                blobs = np.array([])

            if blobs.size > 0:
                for blob in blobs:
                    y, x, r = blob
                    cv2.circle(result_image, (int(x), int(y)), int(r), (0, 255, 0), 1)
                print(f"Detected {len(blobs)} blobs")
            else:
                print("No blobs detected")

            self.preview_plot.setImage(result_image)
        else:
            # Handle other methods
            if method == 'Gaussian Filter':
                result = self.analysis.gaussian_filter(params['Sigma'])
            elif method == 'Median Filter':
                result = self.analysis.median_filter(int(params['Size']))
            elif method == 'Bilateral Filter':
                result = self.analysis.bilateral_filter(params['Sigma Color'], params['Sigma Spatial'])
            elif method == 'Sobel Edge':
                result = self.analysis.sobel_edge()
            elif method == 'Canny Edge':
                result = self.analysis.canny_edge(params['Sigma'])
            elif method == 'Otsu Threshold':
                result = self.analysis.otsu_threshold()
            elif method == 'Adaptive Threshold':
                result = self.analysis.adaptive_threshold(int(params['Block Size']), params['Offset'])
            elif method == 'Erode':
                result = self.analysis.erode(int(params['Size']))
            elif method == 'Dilate':
                result = self.analysis.dilate(int(params['Size']))
            elif method == 'Open':
                result = self.analysis.open(int(params['Size']))
            elif method == 'Close':
                result = self.analysis.close(int(params['Size']))
            elif method == 'Watershed Segmentation':
                result = self.analysis.watershed_segmentation(int(params['Markers']))
            elif method == 'Adjust Gamma':
                result = self.analysis.adjust_gamma(params['Gamma'])
            elif method == 'Equalize Histogram':
                result = self.analysis.equalize_histogram()
            elif method == 'Resize':
                result = self.analysis.resize((int(params['Height']), int(params['Width'])))
            elif method == 'Rotate':
                result = self.analysis.rotate(params['Angle'])
            elif method == 'Unsharp Mask':
                result = self.analysis.unsharp_mask(params['Radius'], params['Amount'])
            elif method == 'Gaussian Gradient Magnitude':
                result = self.analysis.gaussian_gradient_magnitude(params['Sigma'])
            elif method == 'Laplacian':
                result = self.analysis.laplacian()
            elif method == 'Frangi Filter':
                result = self.analysis.frangi(scale_range=(1, 10), scale_step=2,
                                              beta1=params.get('Beta1', 0.5),
                                              beta2=params.get('Beta2', 15))
            elif method == 'Top-hat':
                result = self.analysis.tophat(int(params['Size']))
            elif method == 'Bottom-hat':
                result = self.analysis.bottomhat(int(params['Size']))
            elif method == 'Local Binary Pattern':
                result = self.analysis.local_binary_pattern(P=int(params['P']), R=params['R'])
            elif method == 'Contrast Stretch':
                result = self.analysis.contrast_stretch(in_range=(params['In Min'], params['In Max']))
            elif method == 'Denoise (Non-local Means)':
                result = self.analysis.denoise_nl_means(patch_size=int(params['Patch Size']),
                                                        patch_distance=int(params['Patch Distance']),
                                                        h=params['H'])
            elif method == 'Hessian Matrix Eigenvalues':
                result = self.analysis.hessian_matrix_eigvals(sigma=params['Sigma'])
                if result.ndim > 2:
                    result = result[0]  # Take the first frame for preview
                # Normalize the result for better visualization
                result = (result - np.min(result)) / (np.max(result) - np.min(result))

            else:
                print(f"Unknown method: {method}")
                return

            if isinstance(result, np.ndarray):
                if result.ndim == 2:  # 2D image
                    self.preview_plot.setImage(result)
                elif result.ndim == 3 and result.shape[2] in [3, 4]:  # RGB or RGBA image
                    self.preview_plot.setImage(result)
                elif result.ndim == 3:  # Multiple frames
                    self.preview_plot.setImage(result[0])  # Show the first frame
                else:
                    print(f"Unexpected result shape: {result.shape}")
            else:
                print(f"Unexpected result type: {type(result)}")

        # Update the view
        self.preview_plot.view.autoRange()

    def apply_to_all(self):
        method = self.method_combo.currentText()
        params = self.get_param_values()

        total_frames = self.image.shape[0]
        chunk_size = 10  # Process 10 frames at a time

        progress = QProgressDialog(f"Applying {method} to all frames...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Processing")

        result = np.zeros_like(self.image, dtype=np.float32)
        processed_frames = 0

        def process_chunk():
            nonlocal processed_frames
            chunk_end = min(processed_frames + chunk_size, total_frames)

            temp_analysis = ScikitImageAnalysis()

            for i in range(processed_frames, chunk_end):
                temp_analysis.set_image(self.image[i])

                if method == 'Gaussian Filter':
                    frame_result = temp_analysis.gaussian_filter(sigma=params['Sigma'])
                elif method == 'Median Filter':
                    frame_result = temp_analysis.median_filter(size=int(params['Size']))
                elif method == 'Bilateral Filter':
                    frame_result = temp_analysis.bilateral_filter(sigma_color=params['Sigma Color'], sigma_spatial=params['Sigma Spatial'])
                elif method == 'Sobel Edge':
                    frame_result = temp_analysis.sobel_edge()
                elif method == 'Canny Edge':
                    frame_result = temp_analysis.canny_edge(sigma=params['Sigma'])
                elif method == 'Otsu Threshold':
                    frame_result = temp_analysis.otsu_threshold()
                elif method == 'Adaptive Threshold':
                    frame_result = temp_analysis.adaptive_threshold(block_size=int(params['Block Size']), offset=params['Offset'])
                elif method == 'Erode':
                    frame_result = temp_analysis.erode(size=int(params['Size']))
                elif method == 'Dilate':
                    frame_result = temp_analysis.dilate(size=int(params['Size']))
                elif method == 'Open':
                    frame_result = temp_analysis.open(size=int(params['Size']))
                elif method == 'Close':
                    frame_result = temp_analysis.close(size=int(params['Size']))
                elif method == 'Watershed Segmentation':
                    frame_result = temp_analysis.watershed_segmentation(markers=int(params['Markers']))
                elif method == 'Adjust Gamma':
                    frame_result = temp_analysis.adjust_gamma(gamma=params['Gamma'])
                elif method == 'Equalize Histogram':
                    frame_result = temp_analysis.equalize_histogram()
                elif method == 'Resize':
                    frame_result = temp_analysis.resize((int(params['Height']), int(params['Width'])))
                elif method == 'Rotate':
                    frame_result = temp_analysis.rotate(angle=params['Angle'])
                elif method == 'Unsharp Mask':
                    frame_result = temp_analysis.unsharp_mask(radius=params['Radius'], amount=params['Amount'])
                elif method == 'Gaussian Gradient Magnitude':
                    frame_result = temp_analysis.gaussian_gradient_magnitude(sigma=params['Sigma'])
                elif method == 'Laplacian':
                    frame_result = temp_analysis.laplacian()
                elif method == 'Frangi Filter':
                    frame_result = temp_analysis.frangi(beta1=params['Beta1'], beta2=params['Beta2'])
                elif method == 'Top-hat':
                    frame_result = temp_analysis.tophat(size=int(params['Size']))
                elif method == 'Bottom-hat':
                    frame_result = temp_analysis.bottomhat(size=int(params['Size']))
                elif method == 'Local Binary Pattern':
                    frame_result = temp_analysis.local_binary_pattern(P=int(params['P']), R=params['R'])
                elif method == 'Contrast Stretch':
                    frame_result = temp_analysis.contrast_stretch(in_range=(params['In Min'], params['In Max']))
                elif method == 'Denoise (Non-local Means)':
                    frame_result = temp_analysis.denoise_nl_means(patch_size=int(params['Patch Size']),
                                                                  patch_distance=int(params['Patch Distance']),
                                                                  h=params['H'])
                elif method == 'Hessian Matrix Eigenvalues':
                    frame_result = temp_analysis.hessian_matrix_eigvals(sigma=params['Sigma'])
                else:
                    raise ValueError(f"Unknown method: {method}")

                result[i] = frame_result

                if i % 100 == 0:
                    print(f"Frame {i} - Min: {np.min(result[i])}, Max: {np.max(result[i])}, Mean: {np.mean(result[i])}")

            processed_frames = chunk_end
            progress.setValue(processed_frames)

            if processed_frames < total_frames:
                QTimer.singleShot(0, process_chunk)
            else:
                progress.close()
                self.show_result(result, method)

        QTimer.singleShot(0, process_chunk)

    def show_result(self, result, method):
        print("Processing complete. Result shape:", result.shape)
        print(f"Result stats - Min: {np.min(result)}, Max: {np.max(result)}, Mean: {np.mean(result)}")
        print(f"First frame stats - Min: {np.min(result[0])}, Max: {np.max(result[0])}, Mean: {np.mean(result[0])}")

        # Scale the result to uint16 range
        if result.dtype != np.uint16:
            result_min, result_max = np.min(result), np.max(result)
            if result_min != result_max:
                result_scaled = (result - result_min) / (result_max - result_min)
                result_uint16 = (result_scaled * 65535).astype(np.uint16)
            else:
                result_uint16 = np.zeros_like(result, dtype=np.uint16)
        else:
            result_uint16 = result

        if method in ['Otsu Threshold', 'Adaptive Threshold']:
            result = result.astype(np.uint16) * 65535

        self.analysisCompleted.emit(result_uint16)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Create a sample image for testing
    test_image = np.random.rand(5, 100, 100)  # 5 frames, 100x100 pixels
    console = ScikitAnalysisConsole(test_image)
    console.show()
    sys.exit(app.exec_())
