# flika_bridge_plugin.py

import os
import sys

# Add the directory containing this plugin to the Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

from plugin_base import Plugin
from global_vars import g
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QInputDialog, QAction
from PyQt5.QtCore import pyqtSignal, QObject
import flika
from flika import global_vars as gv
from flika.window import Window as FlikaWindow
from flika.process.file_ import open_file as flika_open_file
import numpy as np
from skimage import io as skio
from flika_compatibility import start_flika, show_flika_error, FLIKA_AVAILABLE


class FlikaMicroViewWindow(QObject):
    timeChanged = pyqtSignal(int)

    def __init__(self, data_or_path):
        super().__init__()
        self.flika_window = None
        self.create_flika_window(data_or_path)
        self.link_time_slider()

    def create_flika_window(self, data_or_path):
        if isinstance(data_or_path, str):
            self.flika_window = flika_open_file(data_or_path)
        elif isinstance(data_or_path, np.ndarray):
            self.flika_window = FlikaWindow(data_or_path)
        else:
            raise ValueError("Invalid input type. Expected file path or numpy array.")

    def link_time_slider(self):
        if hasattr(self.flika_window, 'imageview') and hasattr(self.flika_window.imageview, 'timeLine'):
            self.flika_window.imageview.timeLine.sigPositionChanged.connect(self.on_time_changed)

    def on_time_changed(self):
        current_index = int(self.flika_window.imageview.timeLine.value())
        self.timeChanged.emit(current_index)

    @property
    def image(self):
        return self.flika_window.imageview.image

    @image.setter
    def image(self, new_image):
        self.flika_window.imageview.setImage(new_image)

    def close(self):
        self.flika_window.close()

    @property
    def name(self):
        return self.flika_window.name

class FlikaBridgePlugin(Plugin):
    def __init__(self, microview):
        super().__init__(microview)
        self.name = "Flika Bridge"
        self.version = "1.0.0"
        self.description = "Bridge to FLIKA app"
        self.flika_app = None
        self.flika_windows = []

    def run(self):
        if self.flika_app is None:
            self.initialize_flika()

        self.microview.add_menu_item("Flika", "Open Flika File", self.open_flika_file)
        self.microview.add_menu_item("Flika", "Run Flika Command", self.run_flika_command)
        self.microview.add_menu_item("Flika", "Transfer to Flika", self.transfer_to_flika)
        self.microview.add_menu_item("Flika", "Transfer to MicroView", self.transfer_to_microview)
        self.microview.add_menu_item("Flika", "List Flika Windows", self.list_flika_windows)
        self.microview.add_menu_item("Flika", "Sync Flika Windows", self.sync_flika_windows)

    def initialize_flika(self):
        try:
            self.flika_app = start_flika()
            if FLIKA_AVAILABLE:
                QMessageBox.information(self.microview, "Flika Initialized", "Flika has been successfully initialized.")
            else:
                show_flika_error()
        except Exception as e:
            QMessageBox.critical(self.microview, "Flika Initialization Error", f"Error initializing Flika: {str(e)}")

    def open_flika_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self.microview, "Open File in Flika", "", "All Files (*)")
        if file_path:
            try:
                window = FlikaMicroViewWindow(file_path)
                self.flika_windows.append(window)
                self.microview.add_flika_window(window)
                QMessageBox.information(self.microview, "File Opened", f"File opened in Flika: {file_path}")
            except Exception as e:
                QMessageBox.critical(self.microview, "Error", f"Error opening file in Flika: {str(e)}")

    def sync_flika_windows(self):
        flika_windows = gv.Win.windows
        for flika_window in flika_windows:
            if not any(w.flika_window == flika_window for w in self.flika_windows):
                window = FlikaMicroViewWindow(flika_window.image)
                self.flika_windows.append(window)
                self.microview.add_flika_window(window)
        QMessageBox.information(self.microview, "Sync Complete", f"Synced {len(flika_windows)} Flika windows with MicroView")

    def run_flika_command(self):
        command, ok = QInputDialog.getText(self.microview, "Run Flika Command", "Enter Flika command:")
        if ok and command:
            try:
                exec(command)
                QMessageBox.information(self.microview, "Command Executed", f"Flika command executed: {command}")
            except Exception as e:
                QMessageBox.critical(self.microview, "Error", f"Error executing Flika command: {str(e)}")

    def transfer_to_flika(self):
        if g.m.window_manager.current_window is None:
            QMessageBox.warning(self.microview, "No Window", "No active window in MicroView")
            return

        try:
            microview_image = g.m.window_manager.current_window.image
            flika_window = FlikaWindow(microview_image)
            self.flika_windows.append(flika_window)
            QMessageBox.information(self.microview, "Transfer Complete", "Image transferred to Flika successfully")
        except Exception as e:
            QMessageBox.critical(self.microview, "Transfer Error", f"Error transferring image to Flika: {str(e)}")

    def transfer_to_microview(self):
        if not self.flika_windows:
            QMessageBox.warning(self.microview, "No Flika Windows", "No Flika windows available to transfer")
            return

        window_names = [f"Window {i+1}" for i in range(len(self.flika_windows))]
        window_name, ok = QInputDialog.getItem(self.microview, "Select Flika Window",
                                               "Choose a Flika window to transfer:", window_names, 0, False)
        if ok and window_name:
            index = window_names.index(window_name)
            flika_window = self.flika_windows[index]
            try:
                image = flika_window.image
                new_window = g.m.add_window(image)
                QMessageBox.information(self.microview, "Transfer Complete", "Image transferred to MicroView successfully")
            except Exception as e:
                QMessageBox.critical(self.microview, "Transfer Error", f"Error transferring image to MicroView: {str(e)}")

    def list_flika_windows(self):
        if not self.flika_windows:
            QMessageBox.information(self.microview, "Flika Windows", "No Flika windows are currently open")
        else:
            window_list = "\n".join([f"Window {i+1}: {w.name}" for i, w in enumerate(self.flika_windows)])
            QMessageBox.information(self.microview, "Flika Windows", f"Open Flika windows:\n{window_list}")

    def close(self):
        super().close()
        for window in self.flika_windows:
            try:
                window.close()
            except Exception as e:
                print(f"Error closing Flika window: {str(e)}")
        if self.flika_app is not None:
            try:
                self.flika_app.quit()
            except Exception as e:
                print(f"Error closing Flika app: {str(e)}")
        print("Flika Bridge Plugin closed")

# This line is crucial for the plugin loader to work
Plugin = FlikaBridgePlugin
