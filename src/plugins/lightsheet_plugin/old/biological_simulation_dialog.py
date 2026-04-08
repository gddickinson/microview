#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:37:04 2024

@author: george
"""

# biological_simulation_dialog.py

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QPushButton)

class BiologicalSimulationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Biological Simulation Parameters")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Cell size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Cell Size:"))
        self.size_x = QSpinBox()
        self.size_y = QSpinBox()
        self.size_z = QSpinBox()
        for sb in [self.size_x, self.size_y, self.size_z]:
            sb.setRange(10, 200)
            sb.setValue(100)
            size_layout.addWidget(sb)
        layout.addLayout(size_layout)

        # Cell radius
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Cell Radius:"))
        self.cell_radius = QSpinBox()
        self.cell_radius.setRange(5, 50)
        self.cell_radius.setValue(20)
        radius_layout.addWidget(self.cell_radius)
        layout.addLayout(radius_layout)

        # Number of time points
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Number of Time Points:"))
        self.num_time_points = QSpinBox()
        self.num_time_points.setRange(1, 100)
        self.num_time_points.setValue(10)
        time_layout.addWidget(self.num_time_points)
        layout.addLayout(time_layout)

        # Protein diffusion rate
        diffusion_layout = QHBoxLayout()
        diffusion_layout.addWidget(QLabel("Protein Diffusion Rate:"))
        self.diffusion_rate = QDoubleSpinBox()
        self.diffusion_rate.setRange(0.1, 2.0)
        self.diffusion_rate.setSingleStep(0.1)
        self.diffusion_rate.setValue(0.5)
        diffusion_layout.addWidget(self.diffusion_rate)
        layout.addLayout(diffusion_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect buttons
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_parameters(self):
        return {
            'size': (self.size_z.value(), self.size_y.value(), self.size_x.value()),
            'cell_radius': self.cell_radius.value(),
            'num_volumes': self.num_time_points.value(),
            'diffusion_rate': self.diffusion_rate.value()
        }
