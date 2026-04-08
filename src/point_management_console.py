import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
                             QLineEdit, QColorDialog, QSpinBox, QCheckBox, QListWidget,
                             QLabel, QComboBox, QDialog, QTableView, QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal, QAbstractTableModel
import pyqtgraph as pg

from roi import RectROI

import logging

logger = logging.getLogger(__name__)

class PointSelectionTable(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

class PointSelectionModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data.columns) + 1  # +1 for the 'Selected' column

    def data(self, index, role):
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return 'Yes' if self._data.iloc[index.row()]['selected'] else 'No'
            return str(self._data.iloc[index.row()][self._data.columns[index.column() - 1]])
        elif role == Qt.UserRole:
            return self._data.index[index.row()]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section == 0:
                return 'Selected'
            return self._data.columns[section - 1]

    def setData(self, index, value, role):
        if role == Qt.CheckStateRole and index.column() == 0:
            self._data.at[self._data.index[index.row()], 'selected'] = bool(value)
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def update_data(self, new_data):
        self.beginResetModel()
        self._data = new_data
        self.endResetModel()

class PointManagementConsole(QDialog):
    pointsChanged = pyqtSignal()
    pointSelectionChanged = pyqtSignal(list)  # Emits list of selected point IDs

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.point_data_manager = parent.point_data_manager
        self.roi_operations = parent.roi_operations
        self.initUI()

        # Connect to the data_changed signal of PointDataManager
        self.point_data_manager.data_changed.connect(self.updateSelectionTable)


    def initUI(self):
        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.createAddRemoveTab(), "Add/Remove Points")
        self.tabs.addTab(self.createManipulateTab(), "Manipulate Points")
        self.tabs.addTab(self.createRandomPointsTab(), "Random Points")
        self.tabs.addTab(self.createROIOperationsTab(), "ROI Operations")
        self.tabs.addTab(self.createSelectionTab(), "Point Selection")

        layout.addWidget(self.tabs)

        self.setLayout(layout)
        self.setWindowTitle("Point Management Console")
        self.resize(800, 600)

    def createSelectionTab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Point selection table
        self.selection_table = PointSelectionTable()
        self.selection_model = PointSelectionModel(self.point_data_manager.data)
        self.selection_table.setModel(self.selection_model)
        layout.addWidget(self.selection_table)

        # Buttons for operations on selected points
        button_layout = QHBoxLayout()

        self.create_point_button = QPushButton("Create Point Mode")
        self.create_point_button.setCheckable(True)
        self.create_point_button.toggled.connect(self.toggle_point_creation_mode)
        button_layout.addWidget(self.create_point_button)

        self.delete_point_button = QPushButton("Delete Point Mode")
        self.delete_point_button.setCheckable(True)
        self.delete_point_button.toggled.connect(self.toggle_point_deletion_mode)
        button_layout.addWidget(self.delete_point_button)

        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.deleteSelectedPoints)
        button_layout.addWidget(delete_button)

        change_color_button = QPushButton("Change Color")
        change_color_button.clicked.connect(self.changeSelectedPointsColor)
        button_layout.addWidget(change_color_button)

        select_by_roi_button = QPushButton("Select by ROI")
        select_by_roi_button.clicked.connect(self.selectPointsByROI)
        button_layout.addWidget(select_by_roi_button)

        layout.addLayout(button_layout)

        widget.setLayout(layout)
        return widget


    def deleteSelectedPoints(self):
        selected_indices = self.selection_table.selectionModel().selectedRows()
        selected_ids = [self.selection_model.data(index, Qt.UserRole) for index in selected_indices]
        self.point_data_manager.remove_points(selected_ids)
        self.updateSelectionTable()
        self.pointsChanged.emit()

    def changeSelectedPointsColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            selected_indices = self.selection_table.selectionModel().selectedRows()
            selected_ids = [self.selection_model.data(index, Qt.UserRole) for index in selected_indices]
            self.point_data_manager.change_points_color(selected_ids, color.name())
            self.updateSelectionTable()
            self.pointsChanged.emit()


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

    def selectPointsByROI(self):
        current_window = self.parent.window_management.current_window
        if current_window:
            # Create a temporary rectangular ROI for selection
            temp_roi = RectROI([0, 0], [100, 100], current_window)
            current_window.getView().addItem(temp_roi)

            # Connect ROI change to point selection
            temp_roi.sigRegionChangeFinished.connect(lambda: self.updatePointSelection(temp_roi))

            # Add a button to finish selection
            finish_button = QPushButton("Finish Selection")
            finish_button.clicked.connect(lambda: self.finishROISelection(temp_roi, finish_button))
            current_window.layout().addWidget(finish_button)

    def updatePointSelection(self, roi):
        points_in_roi = self.point_data_manager.get_points_in_roi(roi)
        self.point_data_manager.select_points(points_in_roi.index)
        self.updateSelectionTable()

    def finishROISelection(self, roi, button):
        current_window = self.parent.window_management.current_window
        if current_window:
            current_window.getView().removeItem(roi)
            current_window.layout().removeWidget(button)
            button.deleteLater()
        self.pointsChanged.emit()

    def updateSelectionTable(self):
        self.selection_model.update_data(self.point_data_manager.data)
        print(f"Selection table updated. Total points: {len(self.point_data_manager.data)}")

    def toggle_point_creation_mode(self, checked):
        logger.info(f"toggle_point_creation_mode called with checked={checked}")
        current_window = self.parent.window_management.current_window
        if current_window:
            current_window.set_point_creation_mode(checked)
            if checked:
                self.delete_point_button.setChecked(False)
            logger.info(f"Point creation mode set to {checked} for current window")
        else:
            logger.warning("No current window available to set point creation mode")

    def toggle_point_deletion_mode(self, checked):
        logger.info(f"toggle_point_deletion_mode called with checked={checked}")
        current_window = self.parent.window_management.current_window
        if current_window:
            current_window.set_point_deletion_mode(checked)
            if checked:
                self.create_point_button.setChecked(False)
            logger.info(f"Point deletion mode set to {checked} for current window")
        else:
            logger.warning("No current window available to set point deletion mode")

