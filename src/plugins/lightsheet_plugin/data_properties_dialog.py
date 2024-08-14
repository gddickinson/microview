#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:29:40 2024

@author: george
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QPushButton

class DataPropertiesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.num_z_slices = QSpinBox()
        self.num_z_slices.setRange(1, 1000)
        form.addRow("Number of Z-slices per volume:", self.num_z_slices)

        self.num_channels = QSpinBox()
        self.num_channels.setRange(1, 10)
        form.addRow("Number of channels:", self.num_channels)

        self.pixel_size_xy = QDoubleSpinBox()
        self.pixel_size_xy.setRange(0.01, 1000)
        self.pixel_size_xy.setSuffix(" µm")
        form.addRow("Pixel size (XY):", self.pixel_size_xy)

        self.pixel_size_z = QDoubleSpinBox()
        self.pixel_size_z.setRange(0.01, 1000)
        self.pixel_size_z.setSuffix(" µm")
        form.addRow("Pixel size (Z):", self.pixel_size_z)

        self.slice_angle = QDoubleSpinBox()
        self.slice_angle.setRange(0, 180)
        self.slice_angle.setSuffix(" degrees")
        self.slice_angle.setValue(90)
        form.addRow("Slice angle:", self.slice_angle)

        layout.addLayout(form)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

    def get_properties(self):
        return {
            "num_z_slices": self.num_z_slices.value(),
            "num_channels": self.num_channels.value(),
            "pixel_size_xy": self.pixel_size_xy.value(),
            "pixel_size_z": self.pixel_size_z.value(),
            "slice_angle": self.slice_angle.value()
        }
