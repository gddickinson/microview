#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:50:30 2024

@author: george
"""

# flika_compatibility.py
import numpy as np
from PyQt5.QtWidgets import QWidget
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

try:
    import flika
    from flika.window import Window as FlikaWindow
    from flika.process.file_ import open_file as flika_open_file
    from flika.roi import open_rois as flika_open_rois
    FLIKA_AVAILABLE = True
except ImportError:
    FLIKA_AVAILABLE = False


class MockFlikaWindow:
    def __init__(self, data):
        self.image = data
        self.name = "Mock Flika Window"

    def close(self):
        pass

    def setImage(self, image):
        self.image = image

class FlikaMicroViewWindow(QObject):
    timeChanged = pyqtSignal(int)

    def __init__(self, data_or_path):
        super().__init__()
        self.flika_window = None
        self.create_flika_window(data_or_path)
        self.link_time_slider()

    def create_flika_window(self, data_or_path):
        if FLIKA_AVAILABLE:
            if isinstance(data_or_path, str):
                self.flika_window = flika_open_file(data_or_path)
            elif isinstance(data_or_path, np.ndarray):
                self.flika_window = FlikaWindow(data_or_path)
            else:
                raise ValueError("Invalid input type. Expected file path or numpy array.")
        else:
            if isinstance(data_or_path, str):
                # Use a basic image reading library like PIL or imageio here
                from PIL import Image
                data = np.array(Image.open(data_or_path))
            elif isinstance(data_or_path, np.ndarray):
                data = data_or_path
            else:
                raise ValueError("Invalid input type. Expected file path or numpy array.")
            self.flika_window = MockFlikaWindow(data)

    def link_time_slider(self):
        if FLIKA_AVAILABLE and hasattr(self.flika_window, 'imageview') and hasattr(self.flika_window.imageview, 'timeLine'):
            self.flika_window.imageview.timeLine.sigPositionChanged.connect(self.on_time_changed)

    def on_time_changed(self):
        if FLIKA_AVAILABLE:
            current_index = int(self.flika_window.imageview.timeLine.value())
            self.timeChanged.emit(current_index)

    @property
    def image(self):
        return self.flika_window.image

    @image.setter
    def image(self, new_image):
        self.flika_window.setImage(new_image)

    def close(self):
        self.flika_window.close()

    @property
    def name(self):
        return self.flika_window.name

def start_flika():
    if FLIKA_AVAILABLE:
        return flika.start_flika()
    else:
        print("Flika is not available. Using mock Flika functionality.")
        return None

def show_flika_error():
    if not FLIKA_AVAILABLE:
        QMessageBox.warning(None, "Flika Not Available",
                            "Flika is not installed. Some features may be limited.")




class FlikaCompatibilityLayer:
    @staticmethod
    def open_file(file_path):
        if FLIKA_AVAILABLE:
            return flika_open_file(file_path)
        else:
            raise ImportError("Flika is not available")

    @staticmethod
    def open_rois(file_path):
        if FLIKA_AVAILABLE:
            return flika_open_rois(file_path)
        else:
            raise ImportError("Flika is not available")




class FlikaCompatibleWindow:
    def __init__(self, microview_window):
        self.microview_window = microview_window
        self.image = self.microview_window.image
        self.metadata = {}

    @property
    def imageview(self):
        return self.microview_window

    def close(self):
        self.microview_window.close()

class FlikaCompatibleROI:
    def __init__(self, microview_roi):
        self.microview_roi = microview_roi

    def getTrace(self):
        return self.microview_roi.getTrace()

    @property
    def pos(self):
        return self.microview_roi.pos

    @property
    def size(self):
        return self.microview_roi.size

def flika_open_file(microview_instance, filename):
    microview_instance.loadImage(filename)
    current_window = microview_instance.window_manager.current_window
    return FlikaCompatibleWindow(current_window)

def flika_open_rois(microview_instance, filename):
    microview_rois = microview_instance.open_rois(filename)
    return [FlikaCompatibleROI(roi) for roi in microview_rois]
