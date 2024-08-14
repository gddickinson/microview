# roi.py

import numpy as np
from scipy import ndimage
from skimage import measure
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph as pg

class BaseROI:
    sigROIUpdated = pyqtSignal(object)

    def __init__(self, image_view):
        self.image_view = image_view
        self.create_context_menu()

    def create_context_menu(self):
        self.menu = QMenu()
        self.menu.addAction("Analyze", self.analyze_roi)
        self.menu.addAction("Background Subtraction", self.background_subtraction)
        self.menu.addAction("Remove", self.remove_roi)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.menu.popup(ev.screenPos().toPoint())
        else:
            super().mouseClickEvent(ev)

    def get_roi_data(self):
        image = self.image_view.getImageItem().image
        return self.getArrayRegion(image, self.image_view.getImageItem())

    def analyze_roi(self):
        roi_data = self.get_roi_data()
        self.sigROIUpdated.emit(roi_data)

    def background_subtraction(self):
        try:
            roi_data = self.get_roi_data()
            background = ndimage.median_filter(roi_data, size=3)
            subtracted = roi_data - background
            pg.image(subtracted, title="Background Subtracted")
        except Exception as e:
            print(f"Error in background_subtraction: {str(e)}")

    def remove_roi(self):
        try:
            self.scene().removeItem(self)
            self.sigRemoveRequested.emit(self)
        except Exception as e:
            print(f"Error in remove_roi: {str(e)}")

class RectROI(BaseROI, pg.RectROI):
    def __init__(self, pos, size, image_view, **kwargs):
        pg.RectROI.__init__(self, pos, size, **kwargs)
        BaseROI.__init__(self, image_view)
        self.sigRegionChanged.connect(self.analyze_roi)

class EllipseROI(BaseROI, pg.EllipseROI):
    def __init__(self, pos, size, image_view, **kwargs):
        pg.EllipseROI.__init__(self, pos, size, **kwargs)
        BaseROI.__init__(self, image_view)
        self.sigRegionChanged.connect(self.analyze_roi)

class LineROI(BaseROI, pg.LineSegmentROI):
    def __init__(self, positions, image_view, **kwargs):
        pg.LineSegmentROI.__init__(self, positions, **kwargs)
        BaseROI.__init__(self, image_view)
        self.profile_plot = None
        self.profile_visible = False
        self.profile_curve = None
        self.sigRegionChanged.connect(self.update_profile)
        self.create_context_menu()

        if hasattr(self.image_view, 'timeLine'):
            self.image_view.timeLine.sigPositionChanged.connect(self.update_profile)

    def create_context_menu(self):
        super().create_context_menu()
        self.toggle_profile_action = self.menu.addAction("Show Intensity Profile", self.toggle_intensity_profile)

    def toggle_intensity_profile(self):
        self.profile_visible = not self.profile_visible
        self.update_toggle_profile_text()
        if self.profile_visible:
            self.update_profile()
        elif self.profile_plot:
            self.profile_plot.hide()

    def update_toggle_profile_text(self):
        self.toggle_profile_action.setText("Hide Intensity Profile" if self.profile_visible else "Show Intensity Profile")

    def update_profile(self):
        if self.profile_visible:
            roi_data = self.get_roi_data()
            self.update_intensity_profile(roi_data)
        self.analyze_roi()

    def get_roi_data(self):
        image = self.image_view.getImageItem().image
        if image.ndim == 3:
            current_index = int(self.image_view.timeLine.value())
            image = image[current_index]
        return self.getArrayRegion(image, self.image_view.getImageItem())

    def update_intensity_profile(self, roi_data):
        if roi_data.ndim == 1:
            profile = roi_data
        else:
            profile = np.mean(roi_data, axis=0)

        if self.profile_plot is None:
            self.profile_plot = pg.plot(title="Intensity Profile")
            self.profile_curve = self.profile_plot.plot(pen='r')

        self.profile_curve.setData(profile)
        self.profile_plot.show()

    def remove_roi(self):
        super().remove_roi()
        if self.profile_plot:
            self.profile_plot.close()
