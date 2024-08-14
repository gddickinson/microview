# z_profile.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
import numpy as np

class ZProfileWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.plot = pg.PlotWidget()
        self.plot.setTitle("Z-Profile")
        self.plot.setLabel('left', 'Intensity')
        self.plot.setLabel('bottom', 'Z')
        layout.addWidget(self.plot)

    @pyqtSlot(object, int, int)
    def update_profile(self, image, x, y):
        self.plot.clear()
        if image is None or x is None or y is None:
            self.setVisible(False)
            return

        if image.ndim == 3:
            if 0 <= y < image.shape[1] and 0 <= x < image.shape[2]:
                z_profile = image[:, y, x]
                self.plot.plot(z_profile)
                self.setVisible(True)
            else:
                self.setVisible(False)
        else:
            # For 2D images, we don't have a z-profile
            self.setVisible(False)

    def clear_profile(self):
        self.plot.clear()
        self.plot.setLabel('left', 'Intensity')
        self.plot.setLabel('bottom', 'Z')
        self.setVisible(False)
