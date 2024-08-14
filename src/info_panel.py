#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:07:19 2024

@author: george
"""

# info_panel.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFormLayout
from PyQt5.QtCore import Qt, pyqtSlot
import numpy as np
import pyqtgraph as pg

class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_frame = 0
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        form_layout = QFormLayout()
        layout.addLayout(form_layout)

        self.current_window_label = QLabel("No window selected")
        self.shape_label = QLabel()
        self.dtype_label = QLabel()
        self.mean_label = QLabel()
        self.max_label = QLabel()
        self.min_label = QLabel()
        self.frame_label = QLabel()
        self.mouse_pos_label = QLabel()
        self.intensity_label = QLabel()

        # Set maximum width and word wrap for labels
        max_width = 200  # Adjust this value as needed
        for label in [self.current_window_label, self.shape_label, self.dtype_label,
                      self.mean_label, self.max_label, self.min_label, self.frame_label,
                      self.mouse_pos_label, self.intensity_label]:
            label.setMaximumWidth(max_width)
            label.setWordWrap(True)

        form_layout.addRow("Current Window:", self.current_window_label)
        form_layout.addRow("Shape:", self.shape_label)
        form_layout.addRow("Data Type:", self.dtype_label)
        form_layout.addRow("Mean Intensity:", self.mean_label)
        form_layout.addRow("Max Intensity:", self.max_label)
        form_layout.addRow("Min Intensity:", self.min_label)
        form_layout.addRow("Current Frame:", self.frame_label)
        form_layout.addRow("Mouse Position:", self.mouse_pos_label)
        form_layout.addRow("Intensity at Cursor:", self.intensity_label)


    @pyqtSlot(object)
    def update_info(self, window):
        if window is None:
            self.clear_info()
            return

        image = window.image
        self.current_frame = window.currentIndex if hasattr(window, 'currentIndex') else 0

        self.current_window_label.setText(window.windowTitle())
        self.shape_label.setText(str(image.shape))
        self.dtype_label.setText(str(image.dtype))

        if image.ndim == 3:
            current_image = image[self.current_frame]
        else:
            current_image = image

        self.mean_label.setText(f"{np.mean(current_image):.2f}")
        self.max_label.setText(f"{np.max(current_image):.2f}")
        self.min_label.setText(f"{np.min(current_image):.2f}")
        self.frame_label.setText(str(self.current_frame))




    def clear_info(self):
        self.current_window_label.setText("No window selected")
        self.shape_label.setText("")
        self.dtype_label.setText("")
        self.mean_label.setText("")
        self.max_label.setText("")
        self.min_label.setText("")
        self.frame_label.setText("")
        self.mouse_pos_label.setText("")
        self.intensity_label.setText("")

    @pyqtSlot(int, int)
    def update_mouse_info(self, x, y):
        if x is not None and y is not None:
            self.mouse_pos_label.setText(f"({x}, {y})")
        else:
            self.mouse_pos_label.setText("N/A")

    @pyqtSlot(object)
    def update_intensity(self, intensity):
        if intensity is None:
            self.intensity_label.setText("N/A")
        elif isinstance(intensity, np.ndarray):
            if intensity.size == 1:
                self.intensity_label.setText(f"{intensity.item():.2f}")
            else:
                self.intensity_label.setText(", ".join(f"{value:.2f}" for value in intensity))
        else:
            self.intensity_label.setText(f"{intensity:.2f}")

