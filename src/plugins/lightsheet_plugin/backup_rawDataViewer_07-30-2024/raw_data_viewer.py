# raw_data_viewer.py

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QComboBox
from PyQt5.QtCore import Qt

class RawDataViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.properties = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        self.imageView = pg.ImageView()
        layout.addWidget(self.imageView)

        controlLayout = QHBoxLayout()

        self.volumeSlider = QSlider(Qt.Horizontal)
        self.volumeSlider.valueChanged.connect(self.updateDisplay)
        controlLayout.addWidget(QLabel("Volume:"))
        controlLayout.addWidget(self.volumeSlider)

        self.channelCombo = QComboBox()
        self.channelCombo.currentIndexChanged.connect(self.updateDisplay)
        controlLayout.addWidget(QLabel("Channel:"))
        controlLayout.addWidget(self.channelCombo)

        layout.addLayout(controlLayout)

    def setData(self, data, properties):
        self.data = data
        self.properties = properties

        if self.data is not None and self.properties is not None:
            num_volumes = self.data.shape[0]
            self.volumeSlider.setMaximum(num_volumes - 1)

            self.channelCombo.clear()
            self.channelCombo.addItems([f"Channel {i+1}" for i in range(self.properties['num_channels'])])
            self.channelCombo.addItem("All Channels (Overlay)")

            self.updateDisplay()

    def updateDisplay(self):
        if self.data is None or self.properties is None:
            return

        volume_index = self.volumeSlider.value()
        channel = self.channelCombo.currentIndex()

        num_channels = self.properties['num_channels']
        num_z_slices = self.properties['num_z_slices']

        if channel < num_channels:  # Single channel
            image = self.data[volume_index, channel]
        else:  # All channels overlay
            images = [self.data[volume_index, ch] for ch in range(num_channels)]

            colored_image = np.zeros((*images[0].shape, 3))
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # RGB colors

            for i, img in enumerate(images):
                colored_image += img[:, :, :, np.newaxis] * colors[i % len(colors)]

            colored_image /= colored_image.max()  # Normalize
            image = colored_image

        self.imageView.setImage(image)
        self.imageView.setCurrentIndex(0)  # Reset to first z-slice
