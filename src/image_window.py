import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QDockWidget, QStatusBar, QMenu, QAction, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QCursor
from roi import RectROI, EllipseROI, LineROI
from roi_info import ROIInfoWidget
import logging

logger = logging.getLogger(__name__)

class ImageWindow(QMainWindow):
    timeChanged = pyqtSignal(int)
    windowSelected = pyqtSignal(object)
    roiChanged = pyqtSignal(object)
    pointSelected = pyqtSignal(int)  # Emits point ID
    pointHovered = pyqtSignal(int)  # Emits point ID
    pointCreated = pyqtSignal(float, float)  # Emits x, y coordinates
    pointDeleted = pyqtSignal(int)  # Emits point ID

    def __init__(self, image, title, metadata=None):
        super().__init__()
        self.imageView = pg.ImageView()
        self.setCentralWidget(self.imageView)

        self.image = image
        print(f"ImageWindow init - Image shape: {self.image.shape}, dtype: {self.image.dtype}")
        print(f"Image stats - Min: {np.min(self.image)}, Max: {np.max(self.image)}, Mean: {np.mean(self.image)}")

        self.imageView.setImage(self.image)
        self.setWindowTitle(title)
        self.imageView.timeLine.sigPositionChanged.connect(self.onTimeChanged)
        self.metadata = metadata or {}
        self.currentIndex = 0
        self.is_current = False
        self.installEventFilter(self)
        self.roi = None
        self.rois = []
        self.point_items = []

        # Create and set up the status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Create and add ROI info widget
        self.roi_info_widget = ROIInfoWidget()
        roi_dock = QDockWidget("ROI Info", self)
        roi_dock.setWidget(self.roi_info_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, roi_dock)

        # Update the status bar with metadata information
        self.update_status_bar()

        # Set a default size for the window
        self.resize(800, 600)

        # Add hover label for point info
        self.hover_label = QLabel(self)
        self.hover_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 2px;")
        self.hover_label.hide()

        # Point management modes
        self.point_creation_mode = False
        self.point_deletion_mode = False
        self.point_selection_mode = False

        # Connect mouse events
        self.imageView.scene.sigMouseClicked.connect(self.on_mouse_clicked)
        self.imageView.scene.sigMouseMoved.connect(self.on_mouse_moved)

        logger.debug(f"ImageWindow initialized: {title}")


    def on_mouse_clicked(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.imageView.getImageItem().mapFromScene(event.pos())
            logger.debug(f"Mouse clicked at position: ({pos.x()}, {pos.y()})")
            if self.point_creation_mode:
                logger.info(f"Emitting pointCreated signal with coordinates: ({pos.x()}, {pos.y()})")
                self.pointCreated.emit(pos.x(), pos.y())
            elif self.point_deletion_mode or self.point_selection_mode:
                point_id = self.find_nearest_point(pos.x(), pos.y())
                if point_id is not None:
                    if self.point_deletion_mode:
                        logger.info(f"Emitting pointDeleted signal for point ID: {point_id}")
                        self.pointDeleted.emit(point_id)
                    elif self.point_selection_mode:
                        logger.info(f"Emitting pointSelected signal for point ID: {point_id}")
                        self.pointSelected.emit(point_id)
                else:
                    logger.debug("No nearby point found for deletion or selection")
            else:
                logger.debug("No point management mode is active")


    def on_mouse_moved(self, pos):
        pos = self.imageView.getImageItem().mapFromScene(pos)
        point_id = self.find_nearest_point(pos.x(), pos.y(), max_distance=10)
        if point_id is not None:
            self.showPointInfo(point_id)
            self.pointHovered.emit(point_id)
        else:
            self.hidePointInfo()

    def find_nearest_point(self, x, y, max_distance=5):
        nearest_point = None
        min_distance = float('inf')
        for index, point in enumerate(self.point_items):
            point_pos = point.pos()
            distance = ((point_pos.x() - x)**2 + (point_pos.y() - y)**2)**0.5
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                nearest_point = index
        return nearest_point

    def show_point_info(self, point_id, pos):
        point = self.point_data_manager.data.loc[point_id]
        info = f"ID: {point_id}\nX: {point['x']:.2f}\nY: {point['y']:.2f}\nFrame: {point['frame']}"
        self.hover_label.setText(info)
        self.hover_label.move(pos.toPoint() + QPoint(10, 10))
        self.hover_label.show()

    def hide_point_info(self):
        self.hover_label.hide()

    def onTimeChanged(self):
        self.currentIndex = int(self.imageView.timeLine.value())
        self.timeChanged.emit(self.currentIndex)

    def getImageItem(self):
        return self.imageView.imageItem

    def getView(self):
        view = self.imageView.view
        logger.info(f"Image view range: x={view.viewRange()[0]}, y={view.viewRange()[1]}")
        return view

    def set_as_current(self, is_current):
        self.is_current = is_current
        self.update_border()

    def update_border(self):
        if self.is_current:
            self.setStyleSheet("border: 2px solid green")
        else:
            self.setStyleSheet("")

    def eventFilter(self, obj, event):
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self.windowSelected.emit(self)
        return super().eventFilter(obj, event)

    def add_roi(self, roi_type):
        if roi_type == 'rectangle':
            roi = RectROI([0, 0], [50, 50], self)
        elif roi_type == 'ellipse':
            roi = EllipseROI([0, 0], [50, 50], self)
        elif roi_type == 'line':
            roi = LineROI([[0, 0], [50, 50]], self)
        else:
            raise ValueError(f"Unsupported ROI type: {roi_type}")

        self.rois.append(roi)
        self.getView().addItem(roi)
        roi.sigRegionChanged.connect(lambda: self.roiChanged.emit(roi))
        roi.sigROIUpdated.connect(self.roi_info_widget.update_roi_info)
        roi.sigRemoveRequested.connect(self.remove_roi)

        # Connect the ROI to the time slider if it exists
        if hasattr(self.imageView, 'timeLine') and isinstance(roi, LineROI):
            self.imageView.timeLine.sigPositionChanged.connect(roi.update_profile)

        # Update ROI info immediately after adding
        roi.analyze_roi()

        return roi

    def remove_roi(self, roi):
        if roi in self.rois:
            self.rois.remove(roi)
            self.getView().removeItem(roi)
        if not self.rois:
            self.roiChanged.emit(None)
            self.roi_info_widget.update_roi_info(None)

    def on_roi_changed(self):
        self.roiChanged.emit(self.roi)

    def update_status_bar(self):
        if self.metadata:
            status_text = f"Dims: {self.image.shape} | "
            status_text += f"Dtype: {self.image.dtype} | "
            status_text += f"Pixel Size: {self.metadata.get('pixel_size_um', 'N/A')} µm | "
            status_text += f"Time Interval: {self.metadata.get('time_interval_s', 'N/A')} s"
            self.statusBar.showMessage(status_text)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        # Check if the event position is over an ROI
        pos = self.imageView.view.mapSceneToView(event.pos())
        roi = self.get_roi_at_position(pos)

        if roi:
            # ROI-specific menu
            self.build_roi_menu(menu, roi)
        else:
            # General image menu
            self.build_general_menu(menu)

        menu.exec_(event.globalPos())

    def build_general_menu(self, menu):
        display_metadata_action = QAction("Display Metadata", self)
        display_metadata_action.triggered.connect(self.display_metadata)
        menu.addAction(display_metadata_action)

        # Add ROI creation options
        add_roi_menu = menu.addMenu("Add ROI")
        add_roi_menu.addAction("Rectangle", lambda: self.add_roi('rectangle'))
        add_roi_menu.addAction("Ellipse", lambda: self.add_roi('ellipse'))
        add_roi_menu.addAction("Line", lambda: self.add_roi('line'))

    def build_roi_menu(self, menu, roi):
        analyze_action = QAction("Analyze", self)
        analyze_action.triggered.connect(roi.analyze_roi)
        menu.addAction(analyze_action)

        remove_action = QAction("Remove ROI", self)
        remove_action.triggered.connect(lambda: self.remove_roi(roi))
        menu.addAction(remove_action)

        if isinstance(roi, LineROI):
            toggle_profile_action = QAction("Toggle Intensity Profile", self)
            toggle_profile_action.triggered.connect(roi.toggle_intensity_profile)
            menu.addAction(toggle_profile_action)

    def get_roi_at_position(self, pos):
        for roi in self.rois:
            if roi.contains(pos):
                return roi
        return None

    def display_metadata(self):
        metadata_text = "Image Metadata:\n\n"
        metadata_text += f"Dimensions: {self.image.shape}\n"
        metadata_text += f"Data type: {self.image.dtype}\n"
        for key, value in self.metadata.items():
            metadata_text += f"{key}: {value}\n"
        QMessageBox.information(self, "Image Metadata", metadata_text)

    def setImage(self, image):
        self.image = image
        self.imageView.setImage(image)
        self.update_status_bar()

    def closeEvent(self, event):
        # Emit a signal or call a method to inform the parent that this window is closing
        self.windowSelected.emit(None)
        super().closeEvent(event)

    def plot_points(self, points_data):
        for item in self.point_items:
            self.imageView.view.removeItem(item)
        self.point_items.clear()

        for index, point in points_data.iterrows():
            point_item = pg.ScatterPlotItem([point['x']], [point['y']], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(point['color']))
            point_item.sigClicked.connect(lambda _, points, ev=None, index=index: self.pointSelected.emit(index))
            point_item.sigHovered.connect(lambda _, points, ev=None, index=index: self.showPointInfo(index))
            self.imageView.view.addItem(point_item)
            self.point_items.append(point_item)

    def highlight_points(self, point_ids, color):
        for index, item in enumerate(self.point_items):
            if index in point_ids:
                item.setBrush(pg.mkBrush(color))

    def showPointInfo(self, point_id):
        if hasattr(self, 'point_data_manager'):
            point = self.point_data_manager.data.loc[point_id]
            info = f"ID: {point_id}\nX: {point['x']:.2f}\nY: {point['y']:.2f}\nFrame: {point['frame']}"
            self.hover_label.setText(info)
            cursor_pos = self.imageView.view.mapFromGlobal(QCursor.pos())
            self.hover_label.move(cursor_pos + QPoint(10, 10))
            self.hover_label.show()

    def hidePointInfo(self):
        self.hover_label.hide()

    def set_point_creation_mode(self, mode):
        self.point_creation_mode = mode
        if mode:
            self.point_deletion_mode = False
            self.point_selection_mode = False
        logger.info(f"Point creation mode set to: {mode}")

    def set_point_deletion_mode(self, mode):
        self.point_deletion_mode = mode
        if mode:
            self.point_creation_mode = False
            self.point_selection_mode = False
        logger.info(f"Point deletion mode set to: {mode}")

    def set_point_selection_mode(self, mode):
        self.point_selection_mode = mode
        if mode:
            self.point_creation_mode = False
            self.point_deletion_mode = False
        logger.info(f"Point selection mode set to: {mode}")


    # Getter methods
    def get_image(self):
        return self.image

    def get_metadata(self):
        return self.metadata

    def get_current_frame(self):
        return int(self.imageView.timeLine.value())

    def get_total_frames(self):
        return self.image.shape[0] if self.image.ndim == 3 else 1

    def get_image_item(self):
        return self.imageView.getImageItem()

    def get_view(self):
        return self.imageView.view

    def get_timeline(self):
        return self.imageView.timeLine

    def get_rois(self):
        return self.rois

    def get_image_dimensions(self):
        return self.image.shape

    def get_pixel_size(self):
        return self.metadata.get('pixel_size_um', None)

    def get_time_interval(self):
        return self.metadata.get('time_interval_s', None)

    # setter methods
    def set_metadata(self, new_metadata):
        self.metadata = new_metadata
        self.update_status_bar()
