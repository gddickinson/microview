#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 07:49:18 2024

@author: george
"""

# plugin_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
import os

class Plugin(ABC):
    def __init__(self, microview):
        self.microview = microview
        self.name = "Base Plugin"
        self.version = "1.0.0"
        self.description = "Base plugin class"
        self.widget = None

    @abstractmethod
    def run(self) -> None:
        """Main method to run the plugin."""
        pass

    def close(self):
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.close()

    def setup_plugin_logger(self) -> logging.Logger:
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(os.path.join(logs_dir, f'{self.name}.log'))
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def load_config(self) -> None:
        """Load plugin configuration."""
        # Implement configuration loading logic here
        pass

    def save_config(self) -> None:
        """Save plugin configuration."""
        # Implement configuration saving logic here
        pass

    def cleanup(self) -> None:
        """Clean up resources used by the plugin."""
        pass
