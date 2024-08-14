#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:54:48 2024

@author: george
"""

#synthetic_data_dialog

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg

class SyntheticDataDialog(QDialog):
    dataGenerated = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Data type selection
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            "Empty Array", "Constant Value", "Random Noise",
            "Gaussian Noise", "Calcium Puffs", "Labeled Protein"
        ])
        layout.addWidget(QLabel("Data Type:"))
        layout.addWidget(self.data_type_combo)

        # Dimensions
        dim_layout = QHBoxLayout()
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 1000)
        self.frames_spin.setValue(10)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 1000)
        self.height_spin.setValue(100)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 1000)
        self.width_spin.setValue(100)
        dim_layout.addWidget(QLabel("Frames:"))
        dim_layout.addWidget(self.frames_spin)
        dim_layout.addWidget(QLabel("Height:"))
        dim_layout.addWidget(self.height_spin)
        dim_layout.addWidget(QLabel("Width:"))
        dim_layout.addWidget(self.width_spin)
        layout.addLayout(dim_layout)

        # Additional parameters
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(-1000, 1000)
        layout.addWidget(QLabel("Value:"))
        layout.addWidget(self.value_spin)

        # Generate button
        generate_button = QPushButton("Generate")
        generate_button.clicked.connect(self.generate_data)
        layout.addWidget(generate_button)

        # Preview
        self.preview = pg.ImageView()
        layout.addWidget(self.preview)

        self.setLayout(layout)

    def generate_data(self):
        frames = self.frames_spin.value()
        height = self.height_spin.value()
        width = self.width_spin.value()
        value = self.value_spin.value()

        data_type = self.data_type_combo.currentText()
        if data_type == "Empty Array":
            data = np.zeros((frames, height, width))
        elif data_type == "Constant Value":
            data = np.full((frames, height, width), value)
        elif data_type == "Random Noise":
            data = np.random.random((frames, height, width))
        elif data_type == "Gaussian Noise":
            data = np.random.normal(0, 1, (frames, height, width))
        elif data_type == "Calcium Puffs":
            data = self.generate_calcium_puffs(frames, height, width)
        elif data_type == "Labeled Protein":
            data = self.generate_labeled_protein(frames, height, width)

        self.preview.setImage(data)
        self.dataGenerated.emit(data)

    def generate_calcium_puffs(self, frames, height, width):
        data = np.zeros((frames, height, width))
        num_puffs = np.random.randint(1, 10)
        for _ in range(num_puffs):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            t = np.random.randint(0, frames)
            intensity = np.random.uniform(0.5, 1.0)
            for i in range(max(0, t-5), min(frames, t+6)):
                data[i, max(0, y-2):min(height, y+3), max(0, x-2):min(width, x+3)] += intensity * np.exp(-0.5 * ((i-t)/2)**2)
        return data

    def generate_labeled_protein(self, frames, height, width):
        data = np.zeros((frames, height, width))
        num_proteins = np.random.randint(10, 50)
        for _ in range(num_proteins):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            intensity = np.random.uniform(0.5, 1.0)
            data[:, max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)] = intensity
        return data

    def closeEvent(self, event):
        self.preview.close()
        super().closeEvent(event)
