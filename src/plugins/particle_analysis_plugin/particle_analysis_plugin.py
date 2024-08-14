# particle_analysis_plugin.py

import os, sys

# Add the directory containing this plugin to the Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

import glob
import logging
from logging_config import setup_logging
import traceback
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFileDialog, QProgressBar,
                             QComboBox, QLabel, QGridLayout, QCheckBox, QDoubleSpinBox,
                             QSpinBox, QGroupBox, QScrollArea, QTextEdit, QMessageBox, QApplication)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pandas as pd
import numpy as np
from PyQt5.QtGui import QCursor
from global_vars import g
from analysis_steps import (AddIDsToLocs, LinkPoints, CalculateFeatures, AddNearestNeighbors,
                            AddVelocity, AddMissingPoints, AddBackgroundSubtractedIntensity,
                            FilterTracks, ClassifyTracks, VisualizeTracksCumulative)
from image_utils import ImageHandler, ROIHandler
from microview_utils import MicroViewImageHandler, MicroViewROIHandler

from plugin_base import Plugin
import time


# def setup_logging():
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.DEBUG)

#     file_handler = logging.FileHandler('particle_analysis.log', mode='w')
#     file_handler.setLevel(logging.DEBUG)
#     file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(file_formatter)

#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(logging.INFO)
#     console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
#     console_handler.setFormatter(console_formatter)

#     root_logger.addHandler(file_handler)
#     root_logger.addHandler(console_handler)


#logger = setup_logging()

class Config:
    def __init__(self):
        self.folder_path = ""
        self.pixel_size = 0.108
        self.frame_length = 0.1
        self.max_frames_skipped = 36
        self.max_distance = 3
        self.min_track_length = 4
        self.roi_size = 3
        self.bg_subtraction_method = "local"

class AnalysisWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, config, steps):
        super().__init__()
        self.config = config
        self.steps = steps

    def run(self):
        try:
            data = None
            for i, step in enumerate(self.steps):
                step_name = step.__class__.__name__
                self.log.emit(f"Starting step: {step_name}")
                try:
                    data = step.run(data)
                    progress = int((i + 1) / len(self.steps) * 100)
                    self.progress.emit(progress, f"Completed: {step_name}")
                except Exception as e:
                    error_msg = f"Error in {step_name}: {str(e)}"
                    self.error.emit(error_msg)
                    self.log.emit(f"Error details: {traceback.format_exc()}")
                    return
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.error.emit(error_msg)
            self.log.emit(f"Error details: {traceback.format_exc()}")
        finally:
            self.finished.emit()

class ParticleAnalysisWidget(QWidget):
    def __init__(self, microview):
        super().__init__()
        self.microview = microview
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing ParticleAnalysisWidget")
        self.init_ui()


    def init_ui(self):
        layout = QVBoxLayout()

        # Folder selection
        folder_layout = QGridLayout()
        self.folder_btn = QPushButton("Select Folder")
        self.folder_btn.clicked.connect(self.select_folder)
        self.folder_label = QLabel("No folder selected")
        folder_layout.addWidget(self.folder_btn, 0, 0)
        folder_layout.addWidget(self.folder_label, 0, 1)
        layout.addLayout(folder_layout)

        # Analysis steps selection
        steps_group = QGroupBox("Analysis Steps")
        steps_layout = QVBoxLayout()
        self.step_checkboxes = {}
        for step in [AddIDsToLocs, LinkPoints, CalculateFeatures, AddNearestNeighbors,
                     AddVelocity, AddMissingPoints, AddBackgroundSubtractedIntensity,
                     FilterTracks, ClassifyTracks, VisualizeTracksCumulative]:
            cb = QCheckBox(step.__name__)
            cb.setChecked(True)
            self.step_checkboxes[step.__name__] = cb
            steps_layout.addWidget(cb)
        steps_group.setLayout(steps_layout)
        layout.addWidget(steps_group)

        # Configuration options
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout()
        row = 0

        # Pixel size
        config_layout.addWidget(QLabel("Pixel size (μm):"), row, 0)
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.001, 1000)
        self.pixel_size_spin.setValue(self.config.pixel_size)
        self.pixel_size_spin.valueChanged.connect(lambda v: setattr(self.config, 'pixel_size', v))
        config_layout.addWidget(self.pixel_size_spin, row, 1)
        row += 1

        # Frame length
        config_layout.addWidget(QLabel("Frame length (s):"), row, 0)
        self.frame_length_spin = QDoubleSpinBox()
        self.frame_length_spin.setRange(0.001, 1000)
        self.frame_length_spin.setValue(self.config.frame_length)
        self.frame_length_spin.valueChanged.connect(lambda v: setattr(self.config, 'frame_length', v))
        config_layout.addWidget(self.frame_length_spin, row, 1)
        row += 1

        # Max frames skipped
        config_layout.addWidget(QLabel("Max frames skipped:"), row, 0)
        self.max_frames_skipped_spin = QSpinBox()
        self.max_frames_skipped_spin.setRange(0, 100)
        self.max_frames_skipped_spin.setValue(self.config.max_frames_skipped)
        self.max_frames_skipped_spin.valueChanged.connect(lambda v: setattr(self.config, 'max_frames_skipped', v))
        config_layout.addWidget(self.max_frames_skipped_spin, row, 1)
        row += 1

        # Max distance
        config_layout.addWidget(QLabel("Max distance (pixels):"), row, 0)
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(0.1, 100)
        self.max_distance_spin.setValue(self.config.max_distance)
        self.max_distance_spin.valueChanged.connect(lambda v: setattr(self.config, 'max_distance', v))
        config_layout.addWidget(self.max_distance_spin, row, 1)
        row += 1

        # Min track length
        config_layout.addWidget(QLabel("Min track length:"), row, 0)
        self.min_track_length_spin = QSpinBox()
        self.min_track_length_spin.setRange(2, 1000)
        self.min_track_length_spin.setValue(self.config.min_track_length)
        self.min_track_length_spin.valueChanged.connect(lambda v: setattr(self.config, 'min_track_length', v))
        config_layout.addWidget(self.min_track_length_spin, row, 1)
        row += 1

        # ROI size
        config_layout.addWidget(QLabel("ROI size (pixels):"), row, 0)
        self.roi_size_spin = QSpinBox()
        self.roi_size_spin.setRange(1, 21)
        self.roi_size_spin.setValue(self.config.roi_size)
        self.roi_size_spin.valueChanged.connect(lambda v: setattr(self.config, 'roi_size', v))
        config_layout.addWidget(self.roi_size_spin, row, 1)
        row += 1

        # Background subtraction method
        config_layout.addWidget(QLabel("BG subtraction method:"), row, 0)
        self.bg_subtraction_combo = QComboBox()
        self.bg_subtraction_combo.addItems(["local", "global"])
        self.bg_subtraction_combo.setCurrentText(self.config.bg_subtraction_method)
        self.bg_subtraction_combo.currentTextChanged.connect(lambda v: setattr(self.config, 'bg_subtraction_method', v))
        config_layout.addWidget(self.bg_subtraction_combo, row, 1)

        # Add a text area for logging
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)


        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Run button and progress bar
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        # Make the layout scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setLayout(layout)
        scroll.setWidget(scroll_content)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def check_required_files(self):
        tiff_files = glob.glob(os.path.join(self.config.folder_path, '**/*.tif'), recursive=True)
        if not tiff_files:
            self.show_error("No TIFF files found in the selected folder.")
            return False

        missing_files = []
        for tiff_file in tiff_files:
            points_file = os.path.splitext(tiff_file)[0] + '_locsID.csv'
            if not os.path.exists(points_file):
                missing_files.append(os.path.basename(tiff_file))

        if missing_files:
            self.log("Warning: The following files are missing corresponding _locsID.csv files:", level=logging.WARNING)
            for file in missing_files:
                self.log(f"  - {file}", level=logging.WARNING)
            return False

        return True

    def select_folder(self):
        self.config.folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.folder_label.setText(self.config.folder_path)

    def run_analysis(self):
        if not self.config.folder_path:
            self.show_error("Please select a folder first.")
            return

        self.log_text.clear()
        self.log("Starting analysis...")

        #if not self.check_required_files():
        #    return

        # Check if required files exist
        tiff_files = glob.glob(os.path.join(self.config.folder_path, '**/*.tif'), recursive=True)
        if not tiff_files:
            self.show_error("No TIFF files found in the selected folder.")
            return

        for tiff_file in tiff_files:
            points_file = os.path.splitext(tiff_file)[0] + '_locs.csv'
            if not os.path.exists(points_file):
                self.log(f"Warning: Points file not found for {os.path.basename(tiff_file)}", level=logging.WARNING)

        # Disable UI and show busy cursor
        self.setEnabled(False)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        selected_steps = [step for step, cb in self.step_checkboxes.items() if cb.isChecked()]
        steps = []
        for step in selected_steps:
            step_class = globals()[step]
            if step == 'AddBackgroundSubtractedIntensity':
                steps.append(step_class(self.config, self.microview))
            else:
                steps.append(step_class(self.config))

        self.worker = AnalysisWorker(self.config, steps)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.show_error)
        self.worker.log.connect(self.log)
        self.worker.start()

        self.run_btn.setEnabled(False)
        self.status_label.setText("Analysis in progress...")

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.log(message)

    def analysis_finished(self):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Analysis completed successfully!")
        self.log("Analysis completed successfully!")
        # Re-enable UI and restore cursor
        self.setEnabled(True)
        QApplication.restoreOverrideCursor()

    def show_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_msg}")
        self.log(f"Error: {error_msg}", level=logging.ERROR)
        QMessageBox.critical(self, "Error", error_msg)
        # Re-enable UI and restore cursor in case of error
        self.setEnabled(True)
        QApplication.restoreOverrideCursor()

    def log(self, message, level=logging.INFO):
        logging.log(level, message)
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def analysis_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_msg}")


class ParticleAnalysisPlugin(Plugin):
    def __init__(self, microview):
        super().__init__(microview)
        self.name = "Particle Analysis Plugin"
        self.version = "1.0.0"
        self.description = "Batch particle analysis for csv files with localization info"
        self.widget = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        self.logger.info("Starting Particle Analysis Plugin")
        if self.widget is None:
            self.widget = ParticleAnalysisWidget(self.microview)
        self.widget.show()

    def close(self):
        super().close()
        # Add any specific cleanup code here
        print("Particle Analysis Plugin closed")

# This line is crucial for the plugin loader to work
Plugin = ParticleAnalysisPlugin



if __name__ == '__main__':
    setup_logging()
    # Code to start the plugin in MicroView

    print("Plugin class:", Plugin)
    print("Plugin class attributes:", dir(Plugin))
