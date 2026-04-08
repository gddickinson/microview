# lightsheet_plugin.py

from plugin_interface import PluginInterface
from lightsheetViewer import LightsheetViewer

class Plugin(PluginInterface):
    def __init__(self, parent):
        super().__init__(parent)
        self.name = "LightSheet Viewer"
        self.lightsheet_viewer = None

    def run(self):
        if self.lightsheet_viewer is None:
            self.lightsheet_viewer = LightsheetViewer()
        self.lightsheet_viewer.show()

        # If there's a current window selected in MicroView
        if self.parent.current_window is not None:
            image = self.parent.current_window.image
            if image is not None:
                # Assuming setData is a method in LightsheetViewer that accepts the data
                self.lightsheet_viewer.setData(image)
