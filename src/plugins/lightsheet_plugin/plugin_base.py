#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 07:49:18 2024

@author: george
"""

# plugin_base.py

class Plugin:
    def __init__(self, microview):
        self.microview = microview
        self.name = "Base Plugin"

    def run(self):
        raise NotImplementedError("Subclass must implement abstract method")
