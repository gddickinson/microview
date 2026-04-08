#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:53:30 2024

@author: george
"""

# transformations_dialog

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg
from scipy.ndimage import rotate

class TransformationsDialog(QDialog):
    transformationApplied = pyqtSignal(object)

    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.preview_image = image_data[0] if image_data.ndim == 3 else image_data
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Preview
        self.preview = pg.ImageView()
        self.preview.setImage(self.preview_image)
        layout.addWidget(self.preview)

        # Transformation options
        options_layout = QHBoxLayout()
        self.transform_combo = QComboBox()
        self.transform_combo.addItems(["Rotate 90° CW", "Rotate 90° CCW", "Rotate 180°", "Rotate Custom", "Flip Horizontal", "Flip Vertical"])
        options_layout.addWidget(self.transform_combo)

        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-360, 360)
        self.angle_spin.setEnabled(False)
        options_layout.addWidget(self.angle_spin)

        layout.addLayout(options_layout)

        # Apply button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_transformation)
        layout.addWidget(apply_button)

        self.setLayout(layout)

        # Connect signals
        self.transform_combo.currentIndexChanged.connect(self.on_transform_changed)

    def on_transform_changed(self, index):
        self.angle_spin.setEnabled(index == 3)  # Enable for "Rotate Custom"

    def apply_transformation(self):
        transform = self.transform_combo.currentText()
        if transform == "Rotate 90° CW":
            transformed = np.rot90(self.image_data, k=-1, axes=(-2, -1))
        elif transform == "Rotate 90° CCW":
            transformed = np.rot90(self.image_data, k=1, axes=(-2, -1))
        elif transform == "Rotate 180°":
            transformed = np.rot90(self.image_data, k=2, axes=(-2, -1))
        elif transform == "Rotate Custom":
            angle = self.angle_spin.value()
            transformed = rotate(self.image_data, angle, axes=(-2, -1), reshape=False)
        elif transform == "Flip Horizontal":
            transformed = np.flip(self.image_data, axis=-1)
        elif transform == "Flip Vertical":
            transformed = np.flip(self.image_data, axis=-2)

        self.transformationApplied.emit(transformed)
        self.preview.setImage(transformed[0] if transformed.ndim == 3 else transformed)

    def closeEvent(self, event):
        self.preview.close()
        super().closeEvent(event)
