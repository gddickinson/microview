#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:48:42 2024

@author: george
"""

# base_data_generator.py

from abc import ABC, abstractmethod

class DataGenerator(ABC):
    @abstractmethod
    def generate_volume(self, params):
        pass

    @abstractmethod
    def generate_time_series(self, params):
        pass
