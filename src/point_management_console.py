import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton, 
                             QLineEdit, QColorDialog, QSpinBox, QCheckBox, QListWidget, 
                             QLabel, QComboBox, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

class PointManagementConsole(QDialog):
    pointsChanged = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.point_data_manager = parent.point_data_manager
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.createAddRemoveTab(), "Add/Remove Points")
        self.tabs.addTab(self.createManipulateTab(), "Manipulate Points")
        self.tabs.addTab(self.createRandomPointsTab(), "Random Points")
        self.tabs.addTab(self.createROIOperationsTab(), "ROI Operations")
        
        layout.addWidget(self.tabs)
        
        self.setLayout(layout)
        self.setWindowTitle("Point Management Console")
        self.resize(400, 300)

    def createAddRemoveTab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Add point manually
        addLayout = QHBoxLayout()
        addLayout.addWidget(QLabel("Add Point:"))
        self.xInput = QLineEdit()
        self.yInput = QLineEdit()
        self.frameInput = QLineEdit()
        addLayout.addWidget(QLabel("X:"))
        addLayout.addWidget(self.xInput)
        addLayout.addWidget(QLabel("Y:"))
        addLayout.addWidget(self.yInput)
        addLayout.addWidget(QLabel("Frame:"))
        addLayout.addWidget(self.frameInput)
        addButton = QPushButton("Add")
        addButton.clicked.connect(self.addPoint)
        addLayout.addWidget(addButton)
        layout.addLayout(addLayout)
        
        # Remove point
        removeLayout = QHBoxLayout()
        removeLayout.addWidget(QLabel("Remove Point:"))
        self.removeInput = QLineEdit()
        removeLayout.addWidget(QLabel("Point ID:"))
        removeLayout.addWidget(self.removeInput)
        removeButton = QPushButton("Remove")
        removeButton.clicked.connect(self.removePoint)
        removeLayout.addWidget(removeButton)
        layout.addLayout(removeLayout)
        
        # Point style
        styleLayout = QHBoxLayout()
        styleLayout.addWidget(QLabel("Point Style:"))
        self.sizeInput = QSpinBox()
        self.sizeInput.setRange(1, 20)
        self.sizeInput.setValue(10)
        styleLayout.addWidget(QLabel("Size:"))
        styleLayout.addWidget(self.sizeInput)
        self.colorButton = QPushButton("Color")
        self.colorButton.clicked.connect(self.chooseColor)
        styleLayout.addWidget(self.colorButton)
        layout.addLayout(styleLayout)
        
        widget.setLayout(layout)
        return widget

    def createManipulateTab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Link points
        linkLayout = QHBoxLayout()
        linkLayout.addWidget(QLabel("Link Points:"))
        self.linkInput = QLineEdit()
        linkLayout.addWidget(QLabel("Point IDs (comma-separated):"))
        linkLayout.addWidget(self.linkInput)
        linkButton = QPushButton("Link")
        linkButton.clicked.connect(self.linkPoints)
        linkLayout.addWidget(linkButton)
        layout.addLayout(linkLayout)
        
        # Move point
        moveLayout = QHBoxLayout()
        moveLayout.addWidget(QLabel("Move Point:"))
        self.moveIdInput = QLineEdit()
        self.moveXInput = QLineEdit()
        self.moveYInput = QLineEdit()
        moveLayout.addWidget(QLabel("Point ID:"))
        moveLayout.addWidget(self.moveIdInput)
        moveLayout.addWidget(QLabel("New X:"))
        moveLayout.addWidget(self.moveXInput)
        moveLayout.addWidget(QLabel("New Y:"))
        moveLayout.addWidget(self.moveYInput)
        moveButton = QPushButton("Move")
        moveButton.clicked.connect(self.movePoint)
        moveLayout.addWidget(moveButton)
        layout.addLayout(moveLayout)
        
        widget.setLayout(layout)
        return widget

    def createRandomPointsTab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Number of points
        numLayout = QHBoxLayout()
        numLayout.addWidget(QLabel("Number of Points:"))
        self.numPointsInput = QSpinBox()
        self.numPointsInput.setRange(1, 1000)
        self.numPointsInput.setValue(10)
        numLayout.addWidget(self.numPointsInput)
        layout.addLayout(numLayout)
        
        # Distribution
        distLayout = QHBoxLayout()
        distLayout.addWidget(QLabel("Distribution:"))
        self.distCombo = QComboBox()
        self.distCombo.addItems(["Uniform", "Normal", "Poisson"])
        distLayout.addWidget(self.distCombo)
        layout.addLayout(distLayout)
        
        # Generate button
        genButton = QPushButton("Generate Random Points")
        genButton.clicked.connect(self.generateRandomPoints)
        layout.addWidget(genButton)
        
        widget.setLayout(layout)
        return widget

    def createROIOperationsTab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # ROI selection
        roiLayout = QHBoxLayout()
        roiLayout.addWidget(QLabel("Select ROI:"))
        self.roiCombo = QComboBox()
        self.updateROIList()
        roiLayout.addWidget(self.roiCombo)
        layout.addLayout(roiLayout)
        
        # Operations
        opLayout = QHBoxLayout()
        removeButton = QPushButton("Remove Points in ROI")
        removeButton.clicked.connect(self.removePointsInROI)
        opLayout.addWidget(removeButton)
        moveButton = QPushButton("Move Points in ROI")
        moveButton.clicked.connect(self.movePointsInROI)
        opLayout.addWidget(moveButton)
        layout.addLayout(opLayout)
        
        widget.setLayout(layout)
        return widget

    def addPoint(self):
        x = float(self.xInput.text())
        y = float(self.yInput.text())
        frame = int(self.frameInput.text())
        self.point_data_manager.add_points(np.array([[frame, x, y, 0, frame]]))
        self.pointsChanged.emit()

    def removePoint(self):
        point_id = int(self.removeInput.text())
        self.point_data_manager.remove_points([point_id])
        self.pointsChanged.emit()

    def chooseColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.parent.particle_analysis_operations.options['marker_color'] = color.getRgb()
            self.pointsChanged.emit()

    def linkPoints(self):
        point_ids = [int(id) for id in self.linkInput.text().split(',')]
        self.point_data_manager.link_points(point_ids)
        self.pointsChanged.emit()

    def movePoint(self):
        point_id = int(self.moveIdInput.text())
        new_x = float(self.moveXInput.text())
        new_y = float(self.moveYInput.text())
        self.point_data_manager.move_point(point_id, new_x, new_y)
        self.pointsChanged.emit()

    def generateRandomPoints(self):
        num_points = self.numPointsInput.value()
        distribution = self.distCombo.currentText()
        frame = self.parent.window_management.current_window.currentIndex
        
        if distribution == "Uniform":
            x = np.random.uniform(0, 1000, num_points)
            y = np.random.uniform(0, 1000, num_points)
        elif distribution == "Normal":
            x = np.random.normal(500, 100, num_points)
            y = np.random.normal(500, 100, num_points)
        elif distribution == "Poisson":
            x = np.random.poisson(500, num_points)
            y = np.random.poisson(500, num_points)
        
        points = np.column_stack((np.full(num_points, frame), x, y, np.zeros(num_points), np.full(num_points, frame)))
        self.point_data_manager.add_points(points)
        self.pointsChanged.emit()

    def updateROIList(self):
        self.roiCombo.clear()
        if hasattr(self.parent.window_management.current_window, 'rois'):
            for i, roi in enumerate(self.parent.window_management.current_window.rois):
                self.roiCombo.addItem(f"ROI {i+1}")

    def removePointsInROI(self):
        roi_index = self.roiCombo.currentIndex()
        if roi_index >= 0:
            roi = self.parent.window_management.current_window.rois[roi_index]
            self.point_data_manager.remove_points_in_roi(roi)
            self.pointsChanged.emit()

    def movePointsInROI(self):
        roi_index = self.roiCombo.currentIndex()
        if roi_index >= 0:
            roi = self.parent.window_management.current_window.rois[roi_index]
            dx = 10  # Example: move 10 pixels in x direction
            dy = 10  # Example: move 10 pixels in y direction
            self.point_data_manager.move_points_in_roi(roi, dx, dy)
            self.pointsChanged.emit()

