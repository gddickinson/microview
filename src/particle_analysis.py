#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:17:02 2024

@author: george
"""
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QTableView, QVBoxLayout, QWidget, QDialog, QFormLayout,
                             QLineEdit, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox, QDialogButtonBox, QComboBox,
                             QHBoxLayout, QLabel, QMessageBox, QColorDialog)


from PyQt5.QtCore import QAbstractTableModel, Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import pyqtgraph as pg
from skimage import filters, measure, exposure
import trackpy as tp
from PyQt5.QtGui import QColor

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class ParticleAnalysisResults(QDialog):
    analysisComplete = pyqtSignal(object)

    def __init__(self, parent, image):
        super().__init__(parent)
        self.parent = parent
        self.original_image = image
        self.processed_image = None
        self.df = None
        self.options = {
            'detection_method': 'threshold',
            'min_area': 5,
            'max_area': 500,
            'threshold_method': 'otsu',
            'threshold_factor': 1.0,
            'min_mass': 100,
            'max_mass': 10000,
            'percentile': 95,
            'noise_size': 1,
            'diameter': 11,
            'apply_threshold': True,
            'apply_noise_reduction': True,
            'apply_mass_filter': True,
            'marker_size': 10,
            'marker_color': QColor(255, 0, 0, 120)
        }
        self.current_frame = 0
        self.estimate_initial_parameters()
        self.setup_ui()

    def estimate_initial_parameters(self):
        sample_frame = self.original_image[0] if self.original_image.ndim > 2 else self.original_image
        otsu_threshold = filters.threshold_otsu(sample_frame)
        noise_level = np.std(sample_frame[sample_frame < otsu_threshold])
        self.options['threshold_factor'] = 1.0
        self.options['noise_size'] = int(np.mean([0,noise_level]))
        self.options['min_mass'] = int(otsu_threshold * 5)
        self.options['max_mass'] = int(otsu_threshold * 50)

    def setup_ui(self):
        self.setWindowTitle("Particle Analysis Options")
        main_layout = QHBoxLayout()

        # Options panel
        options_layout = QFormLayout()

        # Add detection method selection
        self.detection_method = QComboBox()
        self.detection_method.addItems(['threshold', 'trackpy'])
        self.detection_method.setCurrentText(self.options['detection_method'])
        self.detection_method.currentTextChanged.connect(self.update_preview)
        options_layout.addRow("Detection Method:", self.detection_method)

        self.min_area = QDoubleSpinBox(minimum=0, maximum=10, value=0.1, decimals=3, singleStep=0.1)
        self.min_area.valueChanged.connect(self.update_preview)
        options_layout.addRow("Min Area:", self.min_area)

        self.max_area = QDoubleSpinBox(minimum=0, maximum=10, value=2.0, decimals=3, singleStep=0.1)
        self.max_area.valueChanged.connect(self.update_preview)
        options_layout.addRow("Max Area:", self.max_area)

        self.threshold_method = QComboBox()
        self.threshold_method.addItems(['otsu', 'local', 'percentile'])
        self.threshold_method.setCurrentText(self.options['threshold_method'])
        options_layout.addRow("Threshold Method:", self.threshold_method)

        self.threshold_factor = QDoubleSpinBox(minimum=0.1, maximum=5.0, singleStep=0.1, value=self.options['threshold_factor'])
        options_layout.addRow("Threshold Factor:", self.threshold_factor)

        self.diameter = QSpinBox(minimum=3, maximum=51, value=3, singleStep=2)
        self.diameter.valueChanged.connect(self.update_preview)
        options_layout.addRow("Particle Diameter:", self.diameter)

        self.min_mass = QDoubleSpinBox(minimum=0, maximum=1000000, value=self.options['min_mass'], decimals=1, singleStep=10)
        self.min_mass.valueChanged.connect(self.update_preview)
        options_layout.addRow("Min Mass:", self.min_mass)

        self.max_mass = QSpinBox(minimum=1, maximum=1000000, value=self.options['max_mass'])
        options_layout.addRow("Max Mass:", self.max_mass)

        self.percentile = QSpinBox(minimum=1, maximum=99, value=64)
        self.percentile.valueChanged.connect(self.update_preview)
        options_layout.addRow("Intensity Percentile:", self.percentile)

        self.noise_size = QSpinBox(minimum=1, maximum=10, value=self.options['noise_size'])
        self.noise_size.valueChanged.connect(self.update_preview)
        options_layout.addRow("Noise Size:", self.noise_size)

        self.smoothing_size = QSpinBox(minimum=3, maximum=15, value=max(3, self.options['noise_size'] + 2))
        self.smoothing_size.setSingleStep(2)  # Ensure only odd values
        self.smoothing_size.valueChanged.connect(self.update_preview)
        options_layout.addRow("Smoothing Size:", self.smoothing_size)

        self.apply_threshold = QCheckBox("Apply Threshold")
        self.apply_threshold.setChecked(self.options['apply_threshold'])
        options_layout.addRow(self.apply_threshold)

        self.apply_noise_reduction = QCheckBox("Apply Noise Reduction")
        self.apply_noise_reduction.setChecked(self.options['apply_noise_reduction'])
        options_layout.addRow(self.apply_noise_reduction)

        self.apply_mass_filter = QCheckBox("Apply Mass Filter")
        self.apply_mass_filter.setChecked(self.options['apply_mass_filter'])
        options_layout.addRow(self.apply_mass_filter)

        # Add checkboxes for optional filtering
        self.apply_area_filter = QCheckBox("Apply Area Filter")
        self.apply_area_filter.setChecked(True)
        self.apply_area_filter.stateChanged.connect(self.update_preview)
        options_layout.addRow(self.apply_area_filter)


        self.apply_linking = QCheckBox("Apply Particle Linking")
        self.apply_linking.setChecked(True)
        options_layout.addRow(self.apply_linking)

        self.apply_trajectory_filter = QCheckBox("Filter Short Trajectories")
        self.apply_trajectory_filter.setChecked(True)
        options_layout.addRow(self.apply_trajectory_filter)

        # Add frame selection
        self.frame_selector = QComboBox()
        self.frame_selector.addItems([str(i) for i in range(self.original_image.shape[0])])
        self.frame_selector.currentIndexChanged.connect(self.update_preview)
        options_layout.addRow("Select Frame:", self.frame_selector)

        # Add marker size option
        self.marker_size = QSpinBox(minimum=1, maximum=50, value=self.options['marker_size'])
        self.marker_size.valueChanged.connect(self.update_preview)
        options_layout.addRow("Marker Size:", self.marker_size)

        # Add marker color option
        self.marker_color_button = QPushButton("Select Marker Color")
        self.marker_color_button.clicked.connect(self.choose_marker_color)
        options_layout.addRow(self.marker_color_button)

        # Connect all option changes to update_preview
        for widget in [self.min_area, self.max_area, self.threshold_factor,
                       self.min_mass, self.max_mass, self.percentile, self.noise_size]:
            widget.valueChanged.connect(self.update_preview)

        self.threshold_method.currentTextChanged.connect(self.update_preview)

        for widget in [self.apply_threshold, self.apply_noise_reduction, self.apply_mass_filter]:
            widget.stateChanged.connect(self.update_preview)

        apply_button = QPushButton("Apply to All Frames")
        apply_button.clicked.connect(self.apply_to_all_frames)
        options_layout.addRow(apply_button)

        main_layout.addLayout(options_layout)

        # Preview panel
        preview_layout = QVBoxLayout()
        self.graphics_layout = pg.GraphicsLayoutWidget()

        # Image with particles
        self.image_view = self.graphics_layout.addViewBox()
        self.image_item = pg.ImageItem()
        self.image_view.addItem(self.image_item)
        self.scatter_plot = pg.ScatterPlotItem(size=self.options['marker_size'],
                                               pen=pg.mkPen(None),
                                               brush=pg.mkBrush(self.options['marker_color']))
        self.image_view.addItem(self.scatter_plot)

        # Colorbar
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.graphics_layout.addItem(self.hist)

        preview_layout.addWidget(QLabel("Processed Image Preview with Detected Particles:"))
        preview_layout.addWidget(self.graphics_layout)

        main_layout.addLayout(preview_layout)

        self.setLayout(main_layout)
        self.update_preview()

    def update_preview(self):
        # Update options from UI
        self.options['detection_method'] = self.detection_method.currentText()
        print(f"Detection method updated to: {self.options['detection_method']}")
        self.options['min_area'] = self.min_area.value()
        self.options['max_area'] = self.max_area.value()
        self.options['threshold_method'] = self.threshold_method.currentText()
        self.options['threshold_factor'] = self.threshold_factor.value()
        self.options['diameter'] = self.diameter.value()
        self.options['min_mass'] = self.min_mass.value()
        self.options['max_mass'] = self.max_mass.value()
        self.options['percentile'] = self.percentile.value()
        self.options['noise_size'] = self.noise_size.value()
        self.options['smoothing_size'] = max(3, self.smoothing_size.value())
        if self.options['smoothing_size'] % 2 == 0:
            self.options['smoothing_size'] += 1
        self.options['apply_threshold'] = self.apply_threshold.isChecked()
        self.options['apply_noise_reduction'] = self.apply_noise_reduction.isChecked()
        self.options['apply_mass_filter'] = self.apply_mass_filter.isChecked()
        self.options['marker_size'] = self.marker_size.value()

        # Get current frame
        current_frame = int(self.frame_selector.currentText())
        frame = self.original_image[current_frame]

        # Process frame
        processed_frame, particles = self.process_frame(frame)

        # Update preview image
        self.image_item.setImage(processed_frame)
        self.image_view.autoRange()

        # Update particle positions
        if particles is not None and not particles.empty:
            self.scatter_plot.setData(particles['centroid-1'], particles['centroid-0'],
                                      size=self.options['marker_size'],
                                      brush=pg.mkBrush(self.options['marker_color']))
            print(f"Updated preview with {len(particles)} particles")
        else:
            self.scatter_plot.clear()
            print("No particles to display in preview")

        # Update histogram
        self.hist.setLevels(processed_frame.min(), processed_frame.max())

    def process_frame(self, frame):
        print(f"Processing frame with method: {self.options['detection_method']}")
        if self.options['detection_method'] == 'threshold':
            return self.process_frame_threshold(frame)
        elif self.options['detection_method'] == 'trackpy':
            return self.process_frame_trackpy(frame)
        else:
            raise ValueError(f"Unknown detection method: {self.options['detection_method']}")

    def process_frame_trackpy(self, frame):
        print("Inside process_frame_trackpy")  # Debug print

        # Convert frame to the correct data type for trackpy
        frame = frame.astype(np.uint8)

        # Calculate diameter and ensure it's an odd integer
        diameter = max(3, int(2 * np.sqrt(self.options['min_area'] / np.pi)))
        if diameter % 2 == 0:
            diameter += 1

        # Set noise_size and smoothing_size
        noise_size = max(1, int(self.options['noise_size']))
        smoothing_size = max(3, noise_size + 2)  # Ensure smoothing_size is larger than noise_size and odd
        if smoothing_size % 2 == 0:
            smoothing_size += 1

        print(f"Using diameter: {diameter}, noise_size: {noise_size}, smoothing_size: {smoothing_size}")  # Debug print

        # Use trackpy for particle detection
        features = tp.locate(frame,
                             diameter=diameter,
                             minmass=self.options['min_mass'],
                             separation=diameter,
                             noise_size=noise_size,
                             smoothing_size=smoothing_size,
                             threshold=self.options['threshold_factor'],
                             percentile=64,  # Lower this value to detect more particles
                             max_iterations=20,  # Increase this for more thorough detection
                             characterize=True)  # Include additional particle properties

        print(f"Trackpy found {len(features)} particles")  # Debug print
        if not features.empty:
            print(f"First 5 particle locations:\n{features[['y', 'x', 'mass', 'size']].head()}")  # Debug print
        else:
            print("No particles found. Consider adjusting parameters.")

         # Convert trackpy results to use 'centroid-0' and 'centroid-1'
        particles = pd.DataFrame({
            'centroid-0': features['y'],
            'centroid-1': features['x'],
            'area': features['size'],
            'mean_intensity': features['mass'] / features['size'],
            'mass': features['mass']
        })

        print(f"Converted {len(particles)} particles for preview")  # Debug print
        if not particles.empty:
            print(f"Area range: {particles['area'].min():.2f} to {particles['area'].max():.2f}")


        # Apply area filter if checkbox is checked
        if self.apply_area_filter.isChecked():
            particles = particles[(particles['area'] >= self.options['min_area']) &
                                  (particles['area'] <= self.options['max_area'])]

            print(f"Area filtered {len(particles)} particles for preview")  # Debug print

        return frame, particles

    def process_frame_threshold(self, frame):
        img = frame.copy()

        if self.options['apply_noise_reduction']:
            img = filters.gaussian(img, sigma=self.options['noise_size'])

        if self.options['apply_threshold']:
            if self.options['threshold_method'] == 'otsu':
                threshold = filters.threshold_otsu(img)
            elif self.options['threshold_method'] == 'local':
                threshold = filters.threshold_local(img, block_size=35, offset=10)
            elif self.options['threshold_method'] == 'percentile':
                threshold = np.percentile(img, self.options['percentile'])

            threshold *= self.options['threshold_factor']
            binary = img > threshold
        else:
            binary = img > 0

        labeled = measure.label(binary)
        props = measure.regionprops_table(labeled, img, properties=['label', 'area', 'centroid', 'mean_intensity'])
        df = pd.DataFrame(props)

        if self.options['apply_mass_filter']:
            df['mass'] = df['area'] * df['mean_intensity']
        # Apply area filter if checkbox is checked
        if self.apply_area_filter.isChecked():
            df = df[(df['area'] >= self.options['min_area']) &
                    (df['area'] <= self.options['max_area']) &
                    (df['mass'] >= self.options['min_mass']) &
                    (df['mass'] <= self.options['max_mass'])]

        # Keep using 'centroid-0' and 'centroid-1'
        particles = pd.DataFrame({
            'centroid-0': df['centroid-0'],
            'centroid-1': df['centroid-1'],
            'area': df['area'],
            'mean_intensity': df['mean_intensity'],
            'mass': df['area'] * df['mean_intensity']
        })

        return img, particles


    def apply_to_all_frames(self):
        print(f"Applying to all frames with method: {self.options['detection_method']}")  # Debug print
        if self.options['detection_method'] == 'trackpy':
            print("Using trackpy batch processing")  # Debug print

            # Use trackpy batch processing
            features = tp.batch(self.original_image,
                                diameter=self.options['diameter'],
                                minmass=self.options['min_mass'],
                                separation=self.options['diameter'],
                                noise_size=self.options['noise_size'],
                                smoothing_size=self.options['smoothing_size'],
                                threshold=self.options['threshold_factor'],
                                percentile=self.options['percentile'],
                                max_iterations=20,
                                characterize=True,
                                processes='auto')

            # Ensure consistent column names
            self.df = pd.DataFrame({
                'frame': features['frame'],
                'y': features['y'],
                'x': features['x'],
                'centroid-0': features['y'],
                'centroid-1': features['x'],
                'area': features['size'],
                'mean_intensity': features['mass'] / features['size'],
                'mass': features['mass']
            })

            # Apply area filter if checkbox is checked
            if self.apply_area_filter.isChecked():
                min_area = self.min_area.value()
                max_area = self.max_area.value()
                self.df = self.df[(self.df['area'] >= min_area) & (self.df['area'] <= max_area)]

            print(f"Particles after area filter: {len(self.df)}")  # Debug print

            # Link particles if checkbox is checked
            if self.apply_linking.isChecked():
                print("Linking particles...")  # Debug print
                search_range = self.options['diameter']
                memory = 3
                linked = tp.link(self.df, search_range=search_range, memory=memory)
                self.df = pd.merge(self.df, linked[['particle']], left_index=True, right_index=True)

                # Filter trajectories if checkbox is checked
                if self.apply_trajectory_filter.isChecked():
                    print("Filtering trajectories...")  # Debug print
                    self.df = tp.filter_stubs(self.df, threshold=5)
            else:
                # If not linking, assign a unique particle ID to each detection
                self.df['particle'] = range(len(self.df))

        else:
            print("Using threshold batch processing")  # Debug print
            # Existing threshold-based method
            results = []
            for frame in range(self.original_image.shape[0]):
                img = self.original_image[frame]
                _, particles = self.process_frame(img)
                particles['frame'] = frame
                results.append(particles)
            self.df = pd.concat(results).reset_index(drop=True)
            # Assign a unique particle ID to each detection
            self.df['particle'] = range(len(self.df))


        print(f"Analysis complete. Found {len(self.df)} particles across all frames.")  # Debug print
        print(f"Columns in final DataFrame: {self.df.columns.tolist()}")  # Debug print
        print(f"First few rows of final DataFrame:\n{self.df.head()}")  # Debug print
        self.analysisComplete.emit(self.df)
        self.accept()

    def run_analysis(self):
        self.df = self.detect_particles()
        if self.options['link_particles']:
            self.df = self.track_particles()

    def detect_particles(self):
        results = []
        for frame in range(self.original_image.shape[0]):
            img = self.original_image[frame]

            # Apply noise reduction
            img_filtered = filters.gaussian(img, sigma=self.options['noise_size'])

            # Apply threshold
            if self.options['threshold_method'] == 'otsu':
                threshold = filters.threshold_otsu(img_filtered)
            elif self.options['threshold_method'] == 'local':
                threshold = filters.threshold_local(img_filtered, block_size=35, offset=10)
            elif self.options['threshold_method'] == 'percentile':
                threshold = np.percentile(img_filtered, self.options['percentile'])

            threshold *= self.options['threshold_factor']

            binary = img_filtered > threshold
            labeled = measure.label(binary)
            props = measure.regionprops_table(labeled, img, properties=['label', 'area', 'centroid', 'mean_intensity'])

            df_frame = pd.DataFrame(props)
            df_frame['mass'] = df_frame['area'] * df_frame['mean_intensity']
            df_frame = df_frame[(df_frame['area'] >= self.options['min_area']) &
                                (df_frame['area'] <= self.options['max_area']) &
                                (df_frame['mass'] >= self.options['min_mass']) &
                                (df_frame['mass'] <= self.options['max_mass'])]
            df_frame['frame'] = frame
            results.append(df_frame)

        self.processed_image = img_filtered  # Store the last processed frame for visualization
        return pd.concat(results).reset_index(drop=True)

    def track_particles(self):
        tp_df = self.df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'})
        linked = tp.link(tp_df, search_range=self.options['search_range'], memory=self.options['memory'])
        return linked

    def show_options_dialog(self):
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Particle Analysis Options")
        layout = QVBoxLayout()

        form_layout = QFormLayout()

        min_area = QSpinBox()
        min_area.setRange(1, 1000)
        min_area.setValue(self.options['min_area'])
        form_layout.addRow("Min Area:", min_area)

        max_area = QSpinBox()
        max_area.setRange(1, 10000)
        max_area.setValue(self.options['max_area'])
        form_layout.addRow("Max Area:", max_area)

        threshold_method = QComboBox()
        threshold_method.addItems(['otsu', 'local', 'percentile'])
        threshold_method.setCurrentText(self.options['threshold_method'])
        form_layout.addRow("Threshold Method:", threshold_method)

        threshold_factor = QDoubleSpinBox()
        threshold_factor.setRange(0.1, 5.0)
        threshold_factor.setSingleStep(0.1)
        threshold_factor.setValue(self.options['threshold_factor'])
        form_layout.addRow("Threshold Factor:", threshold_factor)

        min_mass = QSpinBox()
        min_mass.setRange(1, 100000)
        min_mass.setValue(self.options['min_mass'])
        form_layout.addRow("Min Mass:", min_mass)

        max_mass = QSpinBox()
        max_mass.setRange(1, 1000000)
        max_mass.setValue(self.options['max_mass'])
        form_layout.addRow("Max Mass:", max_mass)

        percentile = QSpinBox()
        percentile.setRange(1, 99)
        percentile.setValue(self.options['percentile'])
        form_layout.addRow("Percentile Threshold:", percentile)

        noise_size = QDoubleSpinBox()
        noise_size.setRange(0.1, 10.0)
        noise_size.setSingleStep(0.1)
        noise_size.setValue(self.options['noise_size'])
        form_layout.addRow("Noise Reduction (sigma):", noise_size)

        show_all_frames = QCheckBox("Show Particles in All Frames")
        show_all_frames.setChecked(self.options['show_all_frames'])
        form_layout.addRow("Display:", show_all_frames)

        link_particles = QCheckBox("Link Particles Between Frames")
        link_particles.setChecked(self.options['link_particles'])
        form_layout.addRow("Tracking:", link_particles)

        search_range = QSpinBox()
        search_range.setRange(1, 50)
        search_range.setValue(self.options['search_range'])
        form_layout.addRow("Search Range:", search_range)

        memory = QSpinBox()
        memory.setRange(0, 10)
        memory.setValue(self.options['memory'])
        form_layout.addRow("Memory:", memory)

        layout.addLayout(form_layout)

        # Add buttons for analysis and visualization
        button_layout = QHBoxLayout()
        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(lambda: self.update_and_analyze(dialog, min_area, max_area, threshold_method, threshold_factor, min_mass, max_mass, percentile, noise_size, show_all_frames, link_particles, search_range, memory))
        button_layout.addWidget(analyze_button)

        show_histogram_button = QPushButton("Show Histogram")
        show_histogram_button.clicked.connect(self.show_histogram)
        button_layout.addWidget(show_histogram_button)

        show_processed_button = QPushButton("Show Processed Image")
        show_processed_button.clicked.connect(self.show_processed_image)
        button_layout.addWidget(show_processed_button)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec_()

    def update_and_analyze(self, dialog, *widgets):
        self.options['min_area'] = widgets[0].value()
        self.options['max_area'] = widgets[1].value()
        self.options['threshold_method'] = widgets[2].currentText()
        self.options['threshold_factor'] = widgets[3].value()
        self.options['min_mass'] = widgets[4].value()
        self.options['max_mass'] = widgets[5].value()
        self.options['percentile'] = widgets[6].value()
        self.options['noise_size'] = widgets[7].value()
        self.options['show_all_frames'] = widgets[8].isChecked()
        self.options['link_particles'] = widgets[9].isChecked()
        self.options['search_range'] = widgets[10].value()
        self.options['memory'] = widgets[11].value()

        self.run_analysis()
        QMessageBox.information(dialog, "Analysis Complete", f"Found {len(self.df)} particles.")

    def show_histogram(self):
        sample_frame = self.original_image[0] if self.original_image.ndim > 2 else self.original_image
        hist, bins = exposure.histogram(sample_frame)

        plt.figure(figsize=(8, 6))
        plt.title("Image Histogram")
        plt.plot(bins, hist)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

    def show_processed_image(self):
        if self.processed_image is not None:
            pg.image(self.processed_image, title="Processed Image")
        else:
            QMessageBox.warning(self.parent, "No Processed Image", "Please run the analysis first.")



    def show_chart(self):
        if self.chart_window is None:
            self.chart_window = QWidget()
            layout = QVBoxLayout()

            # Create table view
            table_view = QTableView()
            model = PandasModel(self.df)
            table_view.setModel(model)
            layout.addWidget(table_view)

            # Create scatter plot
            fig = Figure(figsize=(5, 4), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.scatter(self.df['x'], self.df['y'])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Particle Locations')
            layout.addWidget(canvas)

            self.chart_window.setLayout(layout)
            self.chart_window.setWindowTitle("Particle Analysis Results")
            self.chart_window.resize(600, 400)

        self.chart_window.show()

    def toggle_centroids(self, show):
        if show:
            self.plot_centroids()
        else:
            self.remove_centroids()

    def plot_centroids(self):
        self.remove_centroids()  # Clear existing centroids
        current_frame = self.parent.window_manager.current_window.currentIndex

        if self.options['show_all_frames']:
            df_to_plot = self.df
        else:
            df_to_plot = self.df[self.df['frame'] == current_frame]

        for _, row in df_to_plot.iterrows():
            centroid = pg.ScatterPlotItem([row['centroid-1']], [row['centroid-0']], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0))
            self.parent.window_manager.current_window.view.addItem(centroid)
            self.centroid_items.append(centroid)

    def remove_centroids(self):
        for item in self.centroid_items:
            self.parent.window_manager.current_window.view.removeItem(item)
        self.centroid_items.clear()

    def choose_marker_color(self):
        color = QColorDialog.getColor(initial=self.options['marker_color'], parent=self)
        if color.isValid():
            self.options['marker_color'] = color
            self.update_preview()

    def particle_clicked(self, plot, points):
        if len(points) > 0:
            point = points[0]
            x, y = point.pos()
            particles = self.process_frame(self.original_image[self.current_frame])[1]
            particle = particles[(particles['centroid-1'] == x) & (particles['centroid-0'] == y)].iloc[0]
            properties = f"Area: {particle['area']}\nMass: {particle['mass']}\nMean Intensity: {particle['mean_intensity']}"
            pg.QtGui.QMessageBox.information(self, "Particle Properties", properties)
