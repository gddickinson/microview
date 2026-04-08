#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:43:47 2024

@author: george
"""

# ui_manager.py
import pyqtgraph as pg
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, QLabel,
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QDockWidget, QGridLayout,
                             QScrollArea, QFrame)
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QColorDialog
from PyQt5.QtGui import QColor

class UIManager:
    def __init__(self, parent):
        self.parent = parent
        self.channel_controls = []
        self.channelControlsWidget = None
        self.channelControlsLayout = None  # Add this line

    def setup_ui(self):
        self.create_docks()
        self.create_data_generation_widget()
        self.create_visualization_control_widget()
        self.create_playback_control_widget()
        self.create_blob_detection_widget()
        self.create_channel_controls_dock()

    def create_docks(self):
        self.create_xy_view_dock()
        self.create_xz_view_dock()
        self.create_yz_view_dock()
        self.create_3d_view_dock()
        self.create_blob_visualization_dock()

    def create_xy_view_dock(self):
        self.parent.dockXY = QDockWidget("XY View", self.parent)
        self.parent.imageViewXY = pg.ImageView()
        self.parent.imageViewXY.ui.roiBtn.hide()
        self.parent.imageViewXY.ui.menuBtn.hide()
        self.parent.imageViewXY.setPredefinedGradient('viridis')
        self.parent.imageViewXY.timeLine.sigPositionChanged.connect(self.parent.updateMarkersFromSliders)
        self.parent.dockXY.setWidget(self.parent.imageViewXY)

    def create_xz_view_dock(self):
        self.parent.dockXZ = QDockWidget("XZ View", self.parent)
        self.parent.imageViewXZ = pg.ImageView()
        self.parent.imageViewXZ.ui.roiBtn.hide()
        self.parent.imageViewXZ.ui.menuBtn.hide()
        self.parent.imageViewXZ.setPredefinedGradient('viridis')
        self.parent.imageViewXZ.timeLine.sigPositionChanged.connect(self.parent.updateMarkersFromSliders)
        self.parent.dockXZ.setWidget(self.parent.imageViewXZ)

    def create_yz_view_dock(self):
        self.parent.dockYZ = QDockWidget("YZ View", self.parent)
        self.parent.imageViewYZ = pg.ImageView()
        self.parent.imageViewYZ.ui.roiBtn.hide()
        self.parent.imageViewYZ.ui.menuBtn.hide()
        self.parent.imageViewYZ.setPredefinedGradient('viridis')
        self.parent.imageViewYZ.timeLine.sigPositionChanged.connect(self.parent.updateMarkersFromSliders)
        self.parent.dockYZ.setWidget(self.parent.imageViewYZ)

    def create_3d_view_dock(self):
        self.parent.dock3D = QDockWidget("3D View", self.parent)
        self.parent.glView = self.parent.visualization_manager.create_3d_view()
        self.parent.dock3D.setWidget(self.parent.glView)

    def create_blob_visualization_dock(self):
        self.parent.dockBlobVisualization = QDockWidget("Blob Visualization", self.parent)
        self.parent.blobGLView = self.parent.visualization_manager.create_blob_view()
        self.parent.dockBlobVisualization.setWidget(self.parent.blobGLView)

    def create_data_generation_widget(self):
        self.parent.dockDataGeneration = QDockWidget("Data Generation", self.parent)
        dataGenWidget = QWidget()
        layout = QVBoxLayout(dataGenWidget)

        layout.addWidget(QLabel("Number of Volumes:"))
        self.parent.numVolumesSpinBox = QSpinBox()
        self.parent.numVolumesSpinBox.setRange(1, 100)
        self.parent.numVolumesSpinBox.setValue(10)
        layout.addWidget(self.parent.numVolumesSpinBox)

        layout.addWidget(QLabel("Number of Blobs:"))
        self.parent.numBlobsSpinBox = QSpinBox()
        self.parent.numBlobsSpinBox.setRange(1, 100)
        self.parent.numBlobsSpinBox.setValue(30)
        layout.addWidget(self.parent.numBlobsSpinBox)

        layout.addWidget(QLabel("Noise Level:"))
        self.parent.noiseLevelSpinBox = QDoubleSpinBox()
        self.parent.noiseLevelSpinBox.setRange(0, 1)
        self.parent.noiseLevelSpinBox.setSingleStep(0.01)
        self.parent.noiseLevelSpinBox.setValue(0.02)
        layout.addWidget(self.parent.noiseLevelSpinBox)

        layout.addWidget(QLabel("Movement Speed:"))
        self.parent.movementSpeedSpinBox = QDoubleSpinBox()
        self.parent.movementSpeedSpinBox.setRange(0, 10)
        self.parent.movementSpeedSpinBox.setSingleStep(0.1)
        self.parent.movementSpeedSpinBox.setValue(0.5)
        layout.addWidget(self.parent.movementSpeedSpinBox)

        self.parent.structuredDataCheck = QCheckBox("Generate Structured Data")
        self.parent.structuredDataCheck.setChecked(False)
        layout.addWidget(self.parent.structuredDataCheck)

        self.parent.generateButton = QPushButton("Generate New Data")
        self.parent.generateButton.clicked.connect(self.parent.generateData)
        layout.addWidget(self.parent.generateButton)

        self.parent.saveButton = QPushButton("Save Data")
        self.parent.saveButton.clicked.connect(self.parent.saveData)
        layout.addWidget(self.parent.saveButton)

        self.parent.loadButton = QPushButton("Load Data")
        self.parent.loadButton.clicked.connect(self.parent.loadData)
        layout.addWidget(self.parent.loadButton)

        layout.addStretch(1)
        self.parent.dockDataGeneration.setWidget(dataGenWidget)

    def create_visualization_control_widget(self):
        self.parent.dockVisualizationControl = QDockWidget("Visualization Control", self.parent)
        visControlWidget = QWidget()
        layout = QVBoxLayout(visControlWidget)

        self.setup_channel_controls_widget()
        layout.addWidget(self.channelControlsWidget)

        self.parent.downsamplingCheckBox = QCheckBox("Enable Downsampling")
        self.parent.downsamplingCheckBox.setChecked(False)
        self.parent.downsamplingCheckBox.stateChanged.connect(self.parent.toggleDownsamplingControls)
        self.parent.downsamplingCheckBox.stateChanged.connect(self.parent.updateVisualization)
        layout.addWidget(self.parent.downsamplingCheckBox)

        downsamplingLayout = QHBoxLayout()
        downsamplingLayout.addWidget(QLabel("Max Points:"))
        self.parent.downsamplingSpinBox = QSpinBox()
        self.parent.downsamplingSpinBox.setRange(1000, 1000000)
        self.parent.downsamplingSpinBox.setSingleStep(1000)
        self.parent.downsamplingSpinBox.setValue(100000)
        self.parent.downsamplingSpinBox.valueChanged.connect(self.parent.updateVisualization)
        downsamplingLayout.addWidget(self.parent.downsamplingSpinBox)
        layout.addLayout(downsamplingLayout)

        layout.addWidget(QLabel("Threshold:"))
        self.parent.thresholdSpinBox = QDoubleSpinBox()
        self.parent.thresholdSpinBox.setRange(0, 1)
        self.parent.thresholdSpinBox.setSingleStep(0.1)
        self.parent.thresholdSpinBox.setValue(0.2)
        self.parent.thresholdSpinBox.valueChanged.connect(self.parent.updateThreshold)
        layout.addWidget(self.parent.thresholdSpinBox)

        layout.addWidget(QLabel("Point Size:"))
        self.parent.pointSizeSpinBox = QDoubleSpinBox()
        self.parent.pointSizeSpinBox.setRange(0.1, 10)
        self.parent.pointSizeSpinBox.setSingleStep(0.1)
        self.parent.pointSizeSpinBox.setValue(2)
        self.parent.pointSizeSpinBox.valueChanged.connect(self.parent.updateVisualization)
        layout.addWidget(self.parent.pointSizeSpinBox)

        layout.addWidget(QLabel("3D Rendering Mode:"))
        self.parent.renderModeCombo = QComboBox()
        self.parent.renderModeCombo.addItems(["Points", "Surface", "Wireframe"])
        self.parent.renderModeCombo.currentTextChanged.connect(self.parent.updateRenderMode)
        layout.addWidget(self.parent.renderModeCombo)

        layout.addWidget(QLabel("Color Map:"))
        self.parent.colorMapCombo = QComboBox()
        self.parent.colorMapCombo.addItems(["Viridis", "Plasma", "Inferno", "Magma", "Grayscale"])
        self.parent.colorMapCombo.currentTextChanged.connect(self.parent.updateColorMap)
        layout.addWidget(self.parent.colorMapCombo)

        self.parent.showSliceMarkersCheck = QCheckBox("Show Slice Markers")
        self.parent.showSliceMarkersCheck.stateChanged.connect(self.parent.toggleSliceMarkers)
        layout.addWidget(self.parent.showSliceMarkersCheck)

        layout.addWidget(QLabel("Clip Plane:"))
        self.parent.clipSlider = QSlider(Qt.Horizontal)
        self.parent.clipSlider.setMinimum(0)
        self.parent.clipSlider.setMaximum(100)
        self.parent.clipSlider.setValue(100)
        self.parent.clipSlider.valueChanged.connect(self.parent.updateClipPlane)
        layout.addWidget(self.parent.clipSlider)

        self.parent.syncViewsCheck = QCheckBox("Synchronize 3D Views")
        self.parent.syncViewsCheck.setChecked(False)
        layout.addWidget(self.parent.syncViewsCheck)

        self.parent.autoScaleButton = QPushButton("Auto Scale Views")
        self.parent.autoScaleButton.clicked.connect(self.parent.autoScaleViews)
        layout.addWidget(self.parent.autoScaleButton)

        # Add buttons for orienting views
        self.parent.topDownButton = QPushButton("Top-Down View")
        self.parent.topDownButton.clicked.connect(self.parent.setTopDownView)
        layout.addWidget(self.parent.topDownButton)

        self.parent.sideViewButton = QPushButton("Side View (XZ)")
        self.parent.sideViewButton.clicked.connect(self.parent.setSideView)
        layout.addWidget(self.parent.sideViewButton)

        self.parent.frontViewButton = QPushButton("Front View (YZ)")
        self.parent.frontViewButton.clicked.connect(self.parent.setFrontView)
        layout.addWidget(self.parent.frontViewButton)

        self.intensityProfileButton = QPushButton("Intensity Profile Tool")
        self.intensityProfileButton.setCheckable(True)
        self.intensityProfileButton.toggled.connect(self.parent.toggle_intensity_profile_tool)
        layout.addWidget(self.intensityProfileButton)

        layout.addStretch(1)
        self.parent.dockVisualizationControl.setWidget(visControlWidget)

    def create_channel_controls_dock(self):
        self.dockChannelControls = QDockWidget("Channel Controls", self.parent)

        # Create a scroll area
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)

        # Create a widget to hold the channel controls
        self.channelControlsWidget = QWidget()
        self.channelControlsLayout = QVBoxLayout(self.channelControlsWidget)

        # Add the widget to the scroll area
        scrollArea.setWidget(self.channelControlsWidget)

        # Set the scroll area as the widget for the dock
        self.dockChannelControls.setWidget(scrollArea)

        # Set a minimum width for the dock
        self.dockChannelControls.setMinimumWidth(300)  # Increased from 250 to 300

        # Add the dock to the main window
        self.parent.addDockWidget(Qt.RightDockWidgetArea, self.dockChannelControls)

    def setup_channel_controls_widget(self):
        self.channelControlsWidget = QWidget()
        self.channelControlsLayout = QVBoxLayout(self.channelControlsWidget)
        self.channel_controls = []

    def create_playback_control_widget(self):
        self.parent.dockPlaybackControl = QDockWidget("Playback Control", self.parent)
        playbackControlWidget = QWidget()
        layout = QVBoxLayout(playbackControlWidget)

        layout.addWidget(QLabel("Time:"))
        self.parent.timeSlider = QSlider(Qt.Horizontal)
        self.parent.timeSlider.setMinimum(0)
        self.parent.timeSlider.valueChanged.connect(self.parent.updateTimePoint)
        layout.addWidget(self.parent.timeSlider)

        playbackLayout = QHBoxLayout()
        self.parent.playPauseButton = QPushButton("Play")
        self.parent.playPauseButton.clicked.connect(self.parent.togglePlayback)
        playbackLayout.addWidget(self.parent.playPauseButton)

        self.parent.speedLabel = QLabel("Speed:")
        playbackLayout.addWidget(self.parent.speedLabel)

        self.parent.speedSpinBox = QDoubleSpinBox()
        self.parent.speedSpinBox.setRange(0.1, 10)
        self.parent.speedSpinBox.setSingleStep(0.1)
        self.parent.speedSpinBox.setValue(1)
        self.parent.speedSpinBox.valueChanged.connect(self.parent.updatePlaybackSpeed)
        playbackLayout.addWidget(self.parent.speedSpinBox)

        self.parent.loopCheckBox = QCheckBox("Loop")
        playbackLayout.addWidget(self.parent.loopCheckBox)

        layout.addLayout(playbackLayout)

        layout.addStretch(1)
        self.parent.dockPlaybackControl.setWidget(playbackControlWidget)

    def create_blob_detection_widget(self):
        self.parent.dockBlobDetection = QDockWidget("Blob Detection", self.parent)
        blobDetectionWidget = QWidget()
        layout = QVBoxLayout(blobDetectionWidget)

        layout.addWidget(QLabel("Blob Detection:"))

        blobLayout = QGridLayout()

        blobLayout.addWidget(QLabel("Max Sigma:"), 0, 0)
        self.parent.maxSigmaSpinBox = QDoubleSpinBox()
        self.parent.maxSigmaSpinBox.setRange(1, 100)
        self.parent.maxSigmaSpinBox.setValue(30)
        blobLayout.addWidget(self.parent.maxSigmaSpinBox, 0, 1)

        blobLayout.addWidget(QLabel("Num Sigma:"), 1, 0)
        self.parent.numSigmaSpinBox = QSpinBox()
        self.parent.numSigmaSpinBox.setRange(1, 20)
        self.parent.numSigmaSpinBox.setValue(10)
        blobLayout.addWidget(self.parent.numSigmaSpinBox, 1, 1)

        blobLayout.addWidget(QLabel("Threshold:"), 2, 0)
        self.parent.blobThresholdSpinBox = QDoubleSpinBox()
        self.parent.blobThresholdSpinBox.setRange(0, 1)
        self.parent.blobThresholdSpinBox.setSingleStep(0.01)
        self.parent.blobThresholdSpinBox.setValue(0.5)
        blobLayout.addWidget(self.parent.blobThresholdSpinBox, 2, 1)

        layout.addLayout(blobLayout)

        self.parent.showAllBlobsCheck = QCheckBox("Show All Blobs")
        self.parent.showAllBlobsCheck.setChecked(False)
        self.parent.showAllBlobsCheck.stateChanged.connect(self.parent.updateBlobVisualization)
        layout.addWidget(self.parent.showAllBlobsCheck)

        self.parent.blobDetectionButton = QPushButton("Detect Blobs")
        self.parent.blobDetectionButton.clicked.connect(self.parent.detect_blobs)
        layout.addWidget(self.parent.blobDetectionButton)

        self.parent.showBlobResultsButton = QPushButton("Show Blob Results")
        self.parent.showBlobResultsButton.clicked.connect(self.parent.toggleBlobResults)
        layout.addWidget(self.parent.showBlobResultsButton)

        self.parent.blobAnalysisButton = QPushButton("Analyze Blobs")
        self.parent.blobAnalysisButton.clicked.connect(self.parent.analyzeBlobsasdkjfb)
        layout.addWidget(self.parent.blobAnalysisButton)

        self.parent.timeSeriesButton = QPushButton("Time Series Analysis")
        self.parent.timeSeriesButton.clicked.connect(self.parent.showTimeSeriesAnalysis)
        layout.addWidget(self.parent.timeSeriesButton)

        layout.addStretch(1)
        self.parent.dockBlobDetection.setWidget(blobDetectionWidget)

    def update_channel_controls(self, num_channels, channel_names):
        # Clear existing channel controls
        for i in reversed(range(self.channelControlsLayout.count())):
            item = self.channelControlsLayout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        self.channel_controls.clear()

        for i in range(num_channels):
            channelWidget = QWidget()
            channelLayout = QGridLayout(channelWidget)
            channelLayout.setVerticalSpacing(5)
            channelLayout.setHorizontalSpacing(10)

            channel_name = channel_names[i] if i < len(channel_names) else f'Channel {i+1}'
            channelLayout.addWidget(QLabel(f"{channel_name}:"), 0, 0, 1, 3)

            visibilityCheck = QCheckBox("Visible")
            visibilityCheck.setChecked(True)
            visibilityCheck.stateChanged.connect(self.parent.updateChannelVisibility)
            channelLayout.addWidget(visibilityCheck, 1, 0)

            colorButton = QPushButton()
            colorButton.setFixedSize(20, 20)
            initial_color = self.parent.getChannelColor(i)
            colorButton.setStyleSheet(f"background-color: rgba({initial_color[0]*255}, {initial_color[1]*255}, {initial_color[2]*255}, {initial_color[3]*255})")
            colorButton.clicked.connect(lambda checked, ch=i: self.openColorPicker(ch))
            channelLayout.addWidget(colorButton, 1, 2)

            opacityLabel = QLabel("Opacity:")
            channelLayout.addWidget(opacityLabel, 2, 0)

            opacitySlider = QSlider(Qt.Horizontal)
            opacitySlider.setRange(0, 100)
            opacitySlider.setValue(100)
            opacitySlider.valueChanged.connect(self.parent.updateChannelOpacity)
            channelLayout.addWidget(opacitySlider, 2, 1, 1, 2)

            self.channel_controls.append((visibilityCheck, opacitySlider, colorButton))
            self.channelControlsLayout.addWidget(channelWidget)

            # Add a horizontal line (except for the last channel)
            if i < num_channels - 1:
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                self.channelControlsLayout.addWidget(line)

        # Add a stretch at the end to push all controls to the top
        self.channelControlsLayout.addStretch(1)

    def openColorPicker(self, channel):
        current_color = self.parent.getChannelColor(channel)
        initial_color = QColor.fromRgbF(*current_color)
        color = QColorDialog.getColor(initial=initial_color, options=QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self.parent.setChannelColor(channel, color)
            self.channel_controls[channel][2].setStyleSheet(f"background-color: {color.name()}")
