from PyQt5.QtCore import QObject, pyqtSignal
from image_window import ImageWindow
from flika_compatibility import FlikaMicroViewWindow

class WindowManagementOperations(QObject):
    current_window_changed = pyqtSignal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.current_window = None
        self.windows = []
        self.flika_windows = []

    def add_window(self, window):
        self.windows.append(window)
        window.windowSelected.connect(self.set_current_window)
        self.set_current_window(window)
        window.show()  # Ensure the window is displayed
        return window

    def set_current_window(self, window):
        if self.current_window:
            self.safe_disconnect(self.current_window.imageView.scene.sigMouseMoved, self.parent.update_mouse_position)
            self.safe_disconnect(self.current_window.timeChanged, self.parent.on_time_slider_changed)
            self.safe_disconnect(self.current_window.roiChanged, self.parent.update_roi_info)
            self.current_window.set_as_current(False)

        self.current_window = window
        self.current_window_changed.emit(window)

        if window:
            window.imageView.scene.sigMouseMoved.connect(self.parent.update_mouse_position)
            window.timeChanged.connect(self.parent.on_time_slider_changed)
            window.roiChanged.connect(self.parent.update_roi_info)
            window.set_as_current(True)

        self.parent.update_frame_info(0)
        self.parent.update_roi_info(None)

    def close_current_window(self):
        if self.current_window:
            self.windows.remove(self.current_window)
            self.current_window.close()
            self.current_window = None if not self.windows else self.windows[-1]
            if self.current_window:
                self.set_current_window(self.current_window)
            else:
                self.current_window_changed.emit(None)


    def tile_windows(self):
        # Implement tiling logic here
        pass

    def cascade_windows(self):
        # Implement cascading logic here
        pass

    def safe_disconnect(self, signal, slot):
        try:
            signal.disconnect(slot)
        except TypeError:
            pass
