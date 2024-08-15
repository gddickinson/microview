from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QListWidget, QPushButton, QLabel, QToolBar
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from roi_info import ROIInfoWidget
from roi_z_profile import ROIZProfileWidget
from z_profile import ZProfileWidget
from info_panel import InfoPanel

class UIOperations:
    def __init__(self, parent):
        self.parent = parent

    def createToolbar(self):
        toolbar = QToolBar()
        toolbar.addAction('Undo', self.parent.undo)
        toolbar.addAction('Redo', self.parent.redo)
        toolbar.addSeparator()
        toolbar.addAction('Threshold', self.parent.threshold)
        toolbar.addAction('Measure', self.parent.measure)

        toggle_chart_button = QPushButton("Toggle Results Chart")
        toggle_chart_button.setCheckable(True)
        toggle_chart_button.toggled.connect(self.parent.toggle_results_chart)
        toggle_chart_button.setEnabled(False)

        toggle_centroids_button = QPushButton("Toggle Centroids")
        toggle_centroids_button.setCheckable(True)
        toggle_centroids_button.toggled.connect(self.parent.toggle_centroids)
        toggle_centroids_button.setEnabled(False)

        toolbar.addWidget(toggle_chart_button)
        toolbar.addWidget(toggle_centroids_button)

        return toolbar, toggle_chart_button, toggle_centroids_button

    def createInfoDock(self):
        info_panel = InfoPanel()
        info_dock = QDockWidget("Image Information", self.parent)
        info_dock.setWidget(info_panel)
        info_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        info_dock.setMinimumWidth(250)
        return info_dock, info_panel

    def createZProfileDock(self):
        z_profile_widget = ZProfileWidget()
        z_profile_dock = QDockWidget("Z-Profile", self.parent)
        z_profile_dock.setWidget(z_profile_widget)
        z_profile_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        return z_profile_dock, z_profile_widget

    def createPluginDock(self):
        plugin_widget = QWidget()
        plugin_layout = QVBoxLayout(plugin_widget)
        plugin_list = QListWidget()
        plugin_list.itemDoubleClicked.connect(self.parent.runSelectedPlugin)
        plugin_layout.addWidget(plugin_list)
        run_plugin_button = QPushButton("Run Selected Plugin")
        run_plugin_button.clicked.connect(self.parent.runSelectedPlugin)
        plugin_layout.addWidget(run_plugin_button)

        plugin_dock = QDockWidget("Plugins", self.parent)
        plugin_dock.setWidget(plugin_widget)
        plugin_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        return plugin_dock, plugin_list, run_plugin_button

    def createROIToolsDock(self):
        roi_tools_widget = QWidget()
        roi_tools_layout = QVBoxLayout(roi_tools_widget)

        roi_info_widget = ROIInfoWidget()
        roi_tools_layout.addWidget(roi_info_widget)

        roi_z_profile_widget = ROIZProfileWidget()
        roi_tools_layout.addWidget(roi_z_profile_widget)

        roi_zoom_view = pg.ImageView()
        roi_tools_layout.addWidget(roi_zoom_view)

        roi_tools_dock = QDockWidget("ROI Tools", self.parent)
        roi_tools_dock.setWidget(roi_tools_widget)
        roi_tools_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        return roi_tools_dock, roi_info_widget, roi_z_profile_widget, roi_zoom_view

    def createParticleCountLabel(self):
        return QLabel("Particles in frame: 0")

    def adjustDockSizes(self, width, height):
        top_height = int(height * 0.3)
        self.parent.info_dock.setMaximumHeight(top_height)
        if hasattr(self.parent, 'roi_info_widget'):
            self.parent.roi_info_widget.setMaximumHeight(top_height)

        z_profile_height = int(height * 0.2)
        if hasattr(self.parent, 'z_profile_widget'):
            self.parent.z_profile_widget.setMinimumHeight(z_profile_height)
        if hasattr(self.parent, 'roi_z_profile_widget'):
            self.parent.roi_z_profile_widget.setMinimumHeight(z_profile_height)

        max_info_width = int(width * 0.3)
        self.parent.info_dock.setMaximumWidth(max_info_width)
