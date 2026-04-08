#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:56:41 2024

@author: george
"""

# plugin_interface.py

from abc import ABC, abstractmethod

class PluginInterface(ABC):
    @abstractmethod
    def __init__(self, parent):
        self.parent = parent
        self.name = "Abstract Plugin"

    @abstractmethod
    def run(self):
        pass
