# lightsheet_plugin.py

import os
import sys

# Add the directory containing this plugin to the Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

from plugin_base import Plugin
from lightsheetViewer import LightsheetViewer

class LightsheetPlugin(Plugin):
    def __init__(self, parent):
        super().__init__(parent)
        self.name = "LightSheet Viewer"
        self.lightsheet_viewer = None

    def run(self):
        if self.lightsheet_viewer is None:
            self.lightsheet_viewer = LightsheetViewer()
        self.lightsheet_viewer.show()

        # If there's a current window selected in MicroView
        if self.microview.window_manager.current_window is not None:
            image = self.microview.window_manager.current_window.image
            if image is not None:
                # Assuming setData is a method in LightsheetViewer that accepts the data
                self.lightsheet_viewer.setData(image)

# This line is crucial for the plugin loader to work
Plugin = LightsheetPlugin