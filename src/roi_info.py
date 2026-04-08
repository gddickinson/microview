# roi_info.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot
import numpy as np

class ROIInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Statistic', 'Value'])
        layout.addWidget(self.table)

    @pyqtSlot(object)
    def update_roi_info(self, roi_data):
        self.table.setRowCount(0)
        if roi_data is None:
            return

        stats = {
            'Mean': np.mean(roi_data),
            'Std Dev': np.std(roi_data),
            'Min': np.min(roi_data),
            'Max': np.max(roi_data),
            'Area': roi_data.size
        }

        for i, (stat, value) in enumerate(stats.items()):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(stat))
            self.table.setItem(i, 1, QTableWidgetItem(f"{value:.2f}"))
