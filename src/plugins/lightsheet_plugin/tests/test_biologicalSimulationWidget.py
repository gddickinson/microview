#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:29:14 2024

@author: george
"""

import unittest
from PyQt5.QtWidgets import QApplication
from lightsheetViewer import BiologicalSimulationWidget

class TestBiologicalSimulationWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def setUp(self):
        self.widget = BiologicalSimulationWidget()

    def test_tab_creation(self):
        self.assertEqual(self.widget.tab_widget.count(), 4)
        self.assertEqual(self.widget.tab_widget.tabText(0), "Protein Dynamics")
        self.assertEqual(self.widget.tab_widget.tabText(1), "Cellular Structures")
        self.assertEqual(self.widget.tab_widget.tabText(2), "Organelles")
        self.assertEqual(self.widget.tab_widget.tabText(3), "Calcium Signaling")

    def test_simulation_button(self):
        self.assertTrue(hasattr(self.widget, 'simulate_button'))
        self.assertEqual(self.widget.simulate_button.text(), "Simulate")

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

if __name__ == '__main__':
    unittest.main()
