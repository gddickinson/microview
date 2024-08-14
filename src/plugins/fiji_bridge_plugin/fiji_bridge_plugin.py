#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:45:49 2024

@author: george
"""

# plugins/fiji_bridge/fiji_bridge.py

import os
import sys

# Add the directory containing this plugin to the Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

# plugins/fiji_bridge/fiji_bridge.py

import os
import subprocess
import tempfile
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QPushButton, QVBoxLayout, QWidget, QFileDialog
from plugin_base import Plugin

class FIJIBridge(Plugin):
    def __init__(self, microview):
        super().__init__(microview)
        self.name = "FIJI Bridge"
        self.fiji_path = self.find_fiji()

    def run(self):
        self.widget = QWidget()
        layout = QVBoxLayout()

        launch_button = QPushButton("Launch FIJI")
        launch_button.clicked.connect(self.launch_fiji)
        layout.addWidget(launch_button)

        send_button = QPushButton("Send to FIJI")
        send_button.clicked.connect(self.send_to_fiji)
        layout.addWidget(send_button)

        receive_button = QPushButton("Receive from FIJI")
        receive_button.clicked.connect(self.receive_from_fiji)
        layout.addWidget(receive_button)

        self.widget.setLayout(layout)
        self.widget.show()

    def find_fiji(self):
        # This is a basic implementation. You might need to adjust this
        # based on the typical installation paths on different OS
        possible_paths = [
            "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",  # macOS
            "C:\\Program Files\\Fiji.app\\ImageJ-win64.exe",  # Windows
            "/usr/local/Fiji.app/ImageJ-linux64"  # Linux
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def launch_fiji(self):
        if not self.fiji_path:
            QMessageBox.warning(self.widget, "FIJI Not Found", "Could not find FIJI installation. Please select the FIJI executable.")
            self.fiji_path = QFileDialog.getOpenFileName(self.widget, "Select FIJI Executable")[0]
            if not self.fiji_path:
                return

        try:
            subprocess.Popen([self.fiji_path])
        except Exception as e:
            QMessageBox.warning(self.widget, "Error", f"Failed to launch FIJI: {str(e)}")

    def send_to_fiji(self):
        current_window = self.microview.window_manager.current_window
        if not current_window:
            QMessageBox.warning(self.widget, "Error", "No image selected in MicroView.")
            return

        try:
            image = current_window.image
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
                temp_path = temp_file.name

            # Save the image as a temporary TIFF file
            import tifffile
            tifffile.imwrite(temp_path, image)

            # Create a macro to open the image in FIJI
            macro = f"""
            open("{temp_path}");
            run("Rename...", "title=MicroView_Image");
            """

            with tempfile.NamedTemporaryFile(suffix='.ijm', delete=False) as macro_file:
                macro_file.write(macro.encode('utf-8'))
                macro_path = macro_file.name

            # Run the macro in FIJI
            subprocess.run([self.fiji_path, "-macro", macro_path])

            # Clean up temporary files
            os.unlink(temp_path)
            os.unlink(macro_path)

        except Exception as e:
            QMessageBox.warning(self.widget, "Error", f"Failed to send image to FIJI: {str(e)}")

    def receive_from_fiji(self):
        try:
            # Create a macro to save the current FIJI image
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
                temp_path = temp_file.name

            macro = f"""
            saveAs("Tiff", "{temp_path}");
            """

            with tempfile.NamedTemporaryFile(suffix='.ijm', delete=False) as macro_file:
                macro_file.write(macro.encode('utf-8'))
                macro_path = macro_file.name

            # Run the macro in FIJI
            subprocess.run([self.fiji_path, "-macro", macro_path])

            # Load the saved image into MicroView
            import tifffile
            image = tifffile.imread(temp_path)
            self.microview.loadImage(image, "Image from FIJI")

            # Clean up temporary files
            os.unlink(temp_path)
            os.unlink(macro_path)

        except Exception as e:
            QMessageBox.warning(self.widget, "Error", f"Failed to receive image from FIJI: {str(e)}")
# This line is crucial for the plugin loader to work
Plugin = FIJIBridge
