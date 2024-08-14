#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 08:33:31 2024

@author: george
"""

# window_manager.py
from PyQt5.QtWidgets import QApplication
from flika_compatibility import FlikaMicroViewWindow, FLIKA_AVAILABLE

class WindowManager:
    def __init__(self):
        self.windows = []
        self.current_window = None

    def add_window(self, window):
        self.windows.append(window)
        if not self.current_window:
            self.current_window = window
        window.show()
        self.tile_windows()

    def close_current_window(self):
        if self.current_window:
            if isinstance(self.current_window, FlikaMicroViewWindow):
                self.current_window.close()
            elif hasattr(self.current_window, 'close'):
                self.current_window.close()
            self.windows.remove(self.current_window)
            self.current_window = None if not self.windows else self.windows[-1]

    def tile_windows(self):
        for i, window in enumerate(self.windows):
            screen = QApplication.primaryScreen().geometry()
            width = screen.width() // 2
            height = screen.height() // 2
            x = (i % 2) * width
            y = (i // 2) * height
            window.setGeometry(x, y, width, height)

    def cascade_windows(self):
        for i, window in enumerate(self.windows):
            window.setGeometry(100 + i*20, 100 + i*20, 500, 400)

    def set_current_window(self, window):
        if self.current_window:
            self.current_window.set_as_current(False)
        self.current_window = window
        if window:
            window.set_as_current(True)
