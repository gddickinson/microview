# roi_z_profile.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
import numpy as np

class ROIZProfileWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot = pg.PlotWidget()
        self.plot.setTitle("ROI Z-Profile")
        self.plot.setLabel('left', 'Mean Intensity')
        self.plot.setLabel('bottom', 'Z')
        layout.addWidget(self.plot)

    @pyqtSlot(object, object, object)
    def update_profile(self, image, roi, image_item):
        self.plot.clear()
        if image is None or roi is None or image_item is None:
            return

        try:
            if image.ndim == 3:
                z_profile = []
                for z in range(image.shape[0]):
                    roi_data = roi.getArrayRegion(image[z], image_item)
                    if roi_data is not None and roi_data.size > 0:
                        z_profile.append(np.mean(roi_data))
                    else:
                        z_profile.append(np.nan)

                # Plot only if we have valid data
                if any(not np.isnan(x) for x in z_profile):
                    self.plot.plot(range(len(z_profile)), z_profile, pen='b')
                    self.plot.setLabel('bottom', 'Z-slice')
                else:
                    self.plot.setLabel('bottom', 'No valid data in ROI')
            else:
                # For 2D images, just show the mean intensity as a horizontal line
                roi_data = roi.getArrayRegion(image, image_item)
                if roi_data is not None and roi_data.size > 0:
                    mean_intensity = np.mean(roi_data)
                    self.plot.plot([0, 1], [mean_intensity, mean_intensity], pen='b')
                    self.plot.setLabel('bottom', 'Mean Intensity')
                else:
                    self.plot.setLabel('bottom', 'No valid data in ROI')
        except Exception as e:
            print(f"Error in ROI Z-Profile: {str(e)}")
            self.plot.setLabel('bottom', f'Error: {str(e)}')

    def clear_profile(self):
        self.plot.clear()
        self.plot.setLabel('left', 'Mean Intensity')
        self.plot.setLabel('bottom', 'Z')
