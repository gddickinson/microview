#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:12:24 2024

@author: george
"""

# fluorescence_analysis_plugin.py

import numpy as np
from scipy import stats, ndimage
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QInputDialog,
                             QComboBox, QLabel, QListWidget, QHBoxLayout,
                             QFormLayout, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from plugin_base import Plugin

import logging
import traceback

from scipy import signal

class EventDetector:
    def __init__(self, data):
        self.data = data

    def detect_events_original(self, sensitivity=3.0, threshold=0.02, min_event_duration=20):
        sigma_noise = self.estimate_noise(self.data)
        data_fit, n_states = self.div_segment(self.data, sigma_noise, sensitivity)
        events = self.find_events(data_fit, threshold, min_event_duration)
        return events

    def detect_events_cusum(self, threshold=5, drift=0.05, min_event_duration=20):
        events = []
        cp = self._cusum_detect(self.data, threshold, drift)
        for i in range(len(cp) - 1):
            if cp[i+1] - cp[i] >= min_event_duration:
                events.append((cp[i], min(cp[i+1], len(self.data)-1), self.data[cp[i]], self.data[min(cp[i+1]-1, len(self.data)-1)]))
        return np.array(events)

    def detect_events_threshold(self, threshold=3.5, min_event_duration=20, median_filter_size=5):
        filtered_data = signal.medfilt(self.data, kernel_size=median_filter_size)

        mean = np.mean(filtered_data)
        std = np.std(filtered_data)
        above_threshold = np.abs(filtered_data - mean) > (threshold * std)

        events = []
        event_start = None
        for i, above in enumerate(above_threshold):
            if above and event_start is None:
                event_start = i
            elif not above and event_start is not None:
                if i - event_start >= min_event_duration:
                    events.append((event_start, i, self.data[event_start], self.data[i-1]))
                event_start = None

        if event_start is not None and len(filtered_data) - event_start >= min_event_duration:
            events.append((event_start, len(filtered_data), self.data[event_start], self.data[-1]))

        return np.array(events)

    # ... (include all other methods like estimate_noise, div_segment, etc.)

    def estimate_noise(self, data):
        sorted_wavelet = np.sort(np.abs(np.diff(data) / 1.4))
        return sorted_wavelet[int(round(0.682 * len(sorted_wavelet)))]

    def div_segment(self, data, sigma_noise, sensitivity=2.0, input_type='alpha_value', input_value=0.001):
        n_data = len(data)

        if input_type == 'alpha_value':
            critical_value = stats.t.ppf(1 - input_value / 2, n_data) / sensitivity
        else:
            critical_value = input_value / sensitivity

        centers = np.array([np.mean(data)])
        data_fit = np.full(n_data, centers[0])

        while True:
            new_centers = []
            for center in centers:
                segment = data[data_fit == center]
                if len(segment) < 5:  # Reduce minimum segment length
                    new_centers.append(center)
                    continue

                t_statistic = np.abs(segment - center) / (sigma_noise + 1e-10)
                max_t = np.max(t_statistic)

                if max_t > critical_value:
                    split_point = np.argmax(t_statistic)
                    new_centers.extend([np.mean(segment[:split_point]), np.mean(segment[split_point:])])
                else:
                    new_centers.append(center)

            if len(new_centers) == len(centers):
                break

            centers = np.array(new_centers)
            if len(centers) == 0:
                centers = np.array([np.mean(data)])
            data_fit = centers[np.argmin(np.abs(data[:, np.newaxis] - centers), axis=1)]

        return data_fit, len(centers)

    def find_events(self, data_fit, threshold=0.05, min_event_duration=20):
        unique_states = np.unique(data_fit)
        if len(unique_states) < 2:
            return np.array([])

        state_changes = np.where(np.diff(data_fit) != 0)[0]

        if len(state_changes) == 0:
            return np.array([])

        all_events = []
        current_start = 0
        current_state = data_fit[0]

        for i, change in enumerate(state_changes):
            if change - current_start >= min_event_duration:
                all_events.append([current_start, change, current_state, data_fit[change]])
            current_start = change
            current_state = data_fit[change]

        # Add the last event
        if len(data_fit) - current_start >= min_event_duration:
            all_events.append([current_start, len(data_fit), current_state, data_fit[-1]])

        if not all_events:
            return np.array([])

        all_events = np.array(all_events)

        # Filter events based on threshold
        data_range = np.max(data_fit) - np.min(data_fit)
        normalized_threshold = threshold * data_range
        significant_events = np.abs(all_events[:, 2] - all_events[:, 3]) > normalized_threshold

        return all_events[significant_events]

    def _cusum_detect(self, data, threshold, drift):
        data_normalized = (data - np.mean(data)) / np.std(data)
        s_pos = np.zeros(len(data_normalized))
        s_neg = np.zeros(len(data_normalized))
        change_points = [0]

        for i in range(1, len(data_normalized)):
            s_pos[i] = max(0, s_pos[i-1] + data_normalized[i] - drift)
            s_neg[i] = max(0, s_neg[i-1] - data_normalized[i] - drift)

            if s_pos[i] > threshold or s_neg[i] > threshold:
                change_points.append(i)
                s_pos[i] = 0
                s_neg[i] = 0

        change_points.append(len(data))
        return change_points

class FluorescenceAnalysisPlugin(Plugin):
    def __init__(self, microview):
        super().__init__(microview)
        self.name = "Fluorescence Analysis"
        self.widget = None
        self.rois = []
        self.channel_colors = [(70, 107, 176), (245, 133, 24), (56, 168, 0), (176, 122, 161),
                               (188, 189, 34), (89, 84, 214), (140, 140, 140)]
        self.event_detector = None
        self.params_layout = None
        self.current_roi_index = 0
        self.plot_item = None

    def run(self):
        self.create_widget()
        self.widget.show()

    def create_widget(self):
        self.widget = QWidget()
        layout = QHBoxLayout()

        # Left panel for controls
        left_panel = QVBoxLayout()

        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QListWidget.MultiSelection)
        left_panel.addWidget(QLabel("ROIs:"))
        left_panel.addWidget(self.roi_list)

        self.detection_method_combo = QComboBox()
        self.detection_method_combo.addItems(['Original', 'CUSUM', 'Threshold'])
        left_panel.addWidget(QLabel("Detection Method:"))
        left_panel.addWidget(self.detection_method_combo)

        # Add the ic_combo
        self.ic_combo = QComboBox()
        self.ic_combo.addItems(['BIC_GMM', 'AIC_GMM', 'BIC_RSS'])
        left_panel.addWidget(QLabel("Information Criterion:"))
        left_panel.addWidget(self.ic_combo)

        # Parameters for each method
        self.params_widget = QWidget()
        self.params_layout = QFormLayout()
        self.params_widget.setLayout(self.params_layout)
        left_panel.addWidget(self.params_widget)

        self.detection_method_combo.currentIndexChanged.connect(self.update_params_widget)

        # Add ROI navigation controls
        roi_nav_layout = QHBoxLayout()
        self.roi_selector = QComboBox()
        self.roi_selector.currentIndexChanged.connect(self.on_roi_selected)
        roi_nav_layout.addWidget(self.roi_selector)

        prev_roi_button = QPushButton("Previous ROI")
        prev_roi_button.clicked.connect(self.show_previous_roi)
        roi_nav_layout.addWidget(prev_roi_button)

        next_roi_button = QPushButton("Next ROI")
        next_roi_button.clicked.connect(self.show_next_roi)
        roi_nav_layout.addWidget(next_roi_button)

        left_panel.addLayout(roi_nav_layout)

        analyze_button = QPushButton("Analyze Fluorescence")
        analyze_button.clicked.connect(self.analyze_fluorescence)
        left_panel.addWidget(analyze_button)

        layout.addLayout(left_panel)

        # Right panel for plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        self.widget.setLayout(layout)

        self.update_roi_list()
        self.update_params_widget()

    def update_roi_list(self):
        self.roi_list.clear()
        self.roi_selector.clear()
        self.rois = self.microview.window_manager.current_window.rois
        for i, roi in enumerate(self.rois):
            self.roi_list.addItem(f"ROI {i+1}")
            self.roi_selector.addItem(f"ROI {i+1}")

    def on_roi_selected(self, index):
        self.current_roi_index = index
        self.analyze_current_roi()

    def show_previous_roi(self):
        if self.current_roi_index > 0:
            self.current_roi_index -= 1
            self.roi_selector.setCurrentIndex(self.current_roi_index)
            self.plot_current_roi()

    def show_next_roi(self):
        if self.current_roi_index < len(self.rois) - 1:
            self.current_roi_index += 1
            self.roi_selector.setCurrentIndex(self.current_roi_index)
            self.plot_current_roi()

    def analyze_current_roi(self):
        if self.current_roi_index < len(self.rois):
            roi = self.rois[self.current_roi_index]
            image_data = self.microview.window_manager.current_window.image
            time_series = roi.getArrayRegion(image_data, self.microview.window_manager.current_window.get_image_item(), axes=(1,2)).mean(axis=(1,2))

            logging.info(f"Analyzing ROI {self.current_roi_index}")
            logging.info(f"Time series shape: {time_series.shape}")

            sigma_noise = self.estimate_noise(time_series)
            data_fit, n_states = self.div_segment(time_series, sigma_noise)
            all_data_fits = self.agg_cluster(time_series, data_fit)
            ic_type = self.ic_combo.currentText()
            metrics, n_states = self.compute_ic(time_series, all_data_fits, ic_type)
            best_fit = all_data_fits[:, n_states-1]
            events = self.find_events(best_fit)

            self.plot_results(time_series, data_fit, best_fit, events, self.current_roi_index)
        else:
            logging.error(f"Invalid ROI index: {self.current_roi_index}")

    def plot_current_roi(self):
        self.safe_clear_plot()
        if self.current_roi_index < len(self.rois):
            roi = self.rois[self.current_roi_index]
            image_data = self.microview.window_manager.current_window.image
            logging.info(f"Image data shape: {image_data.shape}")
            if image_data.ndim == 3:  # Ensure we're dealing with a time series
                time_series = roi.getArrayRegion(image_data, self.microview.window_manager.current_window.get_image_item(), axes=(1,2)).mean(axis=(1,2))
                logging.info(f"Time series shape after getArrayRegion: {time_series.shape}")
            else:
                logging.error(f"Unexpected image dimensions: {image_data.shape}")
                return
            self.plot_results(time_series, None, None, None, self.current_roi_index, 1)
        else:
            logging.error(f"Invalid ROI index: {self.current_roi_index}")

    def safe_clear_plot(self):
        try:
            if self.plot_item is not None:
                self.plot_widget.removeItem(self.plot_item)
            self.plot_item = None
        except Exception as e:
            print(f"Error clearing plot: {str(e)}")

    def update_params_widget(self):
        # Clear existing widgets
        for i in reversed(range(self.params_layout.count())):
            widget = self.params_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        method = self.detection_method_combo.currentText()

        if method == 'Original':
            self.params_layout.addRow("Sensitivity:", QDoubleSpinBox(value=3.0, minimum=0.1, maximum=10.0, singleStep=0.1))
            self.params_layout.addRow("Threshold:", QDoubleSpinBox(value=0.02, minimum=0.001, maximum=1.0, singleStep=0.001))
        elif method == 'CUSUM':
            self.params_layout.addRow("Threshold:", QDoubleSpinBox(value=5.0, minimum=0.1, maximum=20.0, singleStep=0.1))
            self.params_layout.addRow("Drift:", QDoubleSpinBox(value=0.05, minimum=0.001, maximum=1.0, singleStep=0.001))
        elif method == 'Threshold':
            self.params_layout.addRow("Threshold:", QDoubleSpinBox(value=3.5, minimum=0.1, maximum=10.0, singleStep=0.1))
            self.params_layout.addRow("Median Filter Size:", QSpinBox(value=5, minimum=1, maximum=21, singleStep=2))

        self.params_layout.addRow("Min Event Duration:", QSpinBox(value=20, minimum=1, maximum=100))


    def analyze_fluorescence(self):
        logging.info("Starting fluorescence analysis")

        selected_items = self.roi_list.selectedItems()
        selected_rois = [self.roi_list.row(item) for item in selected_items]
        if not selected_rois:
            logging.warning("No ROIs selected")
            return

        current_window = self.microview.window_manager.current_window
        if current_window is None:
            logging.error("No image window open")
            return

        image_data = current_window.image
        logging.info(f"Image data shape: {image_data.shape}, dtype: {image_data.dtype}")

        if image_data.ndim != 3:
            logging.error("Image is not 3D (time series)")
            return

        sensitivity, ok = QInputDialog.getDouble(self.widget, "Sensitivity", "Enter sensitivity (default: 1.0, higher is more sensitive):", 1.0, 0.1, 10.0, 2)
        if not ok:
            sensitivity = 1.0

        threshold, ok = QInputDialog.getDouble(self.widget, "Event Threshold", "Enter event threshold (0-1, lower is more sensitive):", 0.1, 0.0, 1.0, 2)
        if not ok:
            threshold = 0.1

        self.safe_clear_plot()

        for i, roi_index in enumerate(selected_rois):

            try:
                roi = self.rois[roi_index]
                logging.info(f"Processing ROI {roi_index}")

                roi_data = roi.getArrayRegion(image_data, current_window.get_image_item(), axes=(1,2))
                logging.info(f"ROI data shape: {roi_data.shape}")

                if roi_data.shape[0] != image_data.shape[0]:
                    logging.error(f"ROI data time points ({roi_data.shape[0]}) do not match image data time points ({image_data.shape[0]})")
                    continue  # Skip this ROI and move to the next one

                time_series = roi_data.mean(axis=(1,2))
                logging.info(f"Time series shape: {time_series.shape}, min: {time_series.min()}, max: {time_series.max()}")

                # Ensure time_series is 1D
                time_series = time_series.squeeze()
                logging.info(f"Squeezed time series shape: {time_series.shape}")

                # Estimate noise
                sigma_noise = self.estimate_noise(time_series)
                logging.info(f"Estimated noise: {sigma_noise}")

                # Perform divisive segmentation
                data_fit, n_states = self.div_segment(time_series, sigma_noise, sensitivity=sensitivity)
                logging.info(f"Divisive segmentation complete. Number of states: {n_states}")
                logging.info(f"Unique states in data_fit: {np.unique(data_fit)}")

                # perform agglomerative clustering
                all_data_fits = self.agg_cluster(time_series, data_fit)
                logging.info(f"Agglomerative clustering complete. Shape of all_data_fits: {all_data_fits.shape}")
                logging.info(f"Unique states in each column of all_data_fits: {[np.unique(all_data_fits[:, i]) for i in range(all_data_fits.shape[1])]}")

                # Compute information criterion
                ic_type = self.ic_combo.currentText()
                metrics, n_states = self.compute_ic(time_series, all_data_fits, ic_type)
                logging.info(f"Information criterion computed. Best number of states: {n_states}")

                # Get the best fit
                best_fit = all_data_fits[:, n_states-1]
                logging.info(f"Best fit shape: {best_fit.shape}")

                # Find events
                method = self.detection_method_combo.currentText()
                params = {}

                logging.debug(f"Number of items in params_layout: {self.params_layout.count()}")
                for i in range(self.params_layout.count()):
                    item = self.params_layout.itemAt(i)
                    if item and item.widget():
                        logging.debug(f"Item {i}: {item.widget().objectName()}")
                    else:
                        logging.debug(f"Item {i} is None or has no widget")

                for i in range(self.params_layout.rowCount()):
                    label_item = self.params_layout.itemAt(i*2)
                    value_item = self.params_layout.itemAt(i*2+1)
                    if label_item and value_item:
                        label = label_item.widget().text().replace(":", "").lower().replace(" ", "_")
                        value = value_item.widget().value()
                        params[label] = value

                self.event_detector = EventDetector(time_series)

                if method == 'Original':
                    events = self.event_detector.detect_events_original(**params)
                elif method == 'CUSUM':
                    events = self.event_detector.detect_events_cusum(**params)
                elif method == 'Threshold':
                    events = self.event_detector.detect_events_threshold(**params)

                logging.info(f"Number of events found: {len(events)}")

                # Instead of plotting all ROIs at once, just update the current ROI
                if i == self.current_roi_index:
                    self.plot_results(time_series, data_fit, best_fit, events, i, 1)
                    logging.info("Results plotted successfully")

            except Exception as e:
                logging.error(f"Error processing ROI {roi_index}: {str(e)}")
                logging.error(traceback.format_exc())

        self.plot_widget.show()
        logging.info("Fluorescence analysis complete")

    def estimate_noise(self, data, option=1):
        if option == 1:
            # Option 1: Adapted from STaSI
            sorted_wavelet = np.sort(np.abs(np.diff(data) / 1.4))
            sigma_noise = sorted_wavelet[int(round(0.682 * len(sorted_wavelet)))]
        else:
            # Option 2: Adapted from OWLET_denoise (requires PyWavelets)
            import pywt
            level = int(np.log2(len(data)))
            coeffs = pywt.wavedec(data, 'haar', level=level)
            sigma_noise = np.median(np.abs(coeffs[-1])) / 0.6745

        return sigma_noise

    def div_segment(self, data, sigma_noise, sensitivity=2.0, input_type='alpha_value', input_value=0.01):
        n_data = len(data)

        if input_type == 'alpha_value':
            critical_value = stats.t.ppf(1 - input_value / 2, n_data) / sensitivity
        else:
            critical_value = input_value / sensitivity

        centers = np.array([np.nanmean(data)])
        data_fit = np.full(n_data, centers[0])

        while True:
            new_centers = []
            for center in centers:
                segment = data[data_fit == center]
                if len(segment) < 5:  # Reduce minimum segment length
                    new_centers.append(center)
                    continue

                t_statistic = np.abs(segment - center) / (sigma_noise + 1e-10)
                max_t = np.nanmax(t_statistic)

                if max_t > critical_value:
                    split_point = np.nanargmax(t_statistic)
                    new_centers.extend([np.nanmean(segment[:split_point]), np.nanmean(segment[split_point:])])
                else:
                    new_centers.append(center)

            if len(new_centers) == len(centers):
                break

            centers = np.array(new_centers)
            if len(centers) == 0:
                centers = np.array([np.nanmean(data)])
            data_fit = centers[np.nanargmin(np.abs(data[:, np.newaxis] - centers), axis=1)]

        return data_fit, len(centers)

    def detect_cps(self, data, input_type='alpha_value', input_value=0.05, min_data_points=2):
        n_data_points = len(data)

        if input_type == 'alpha_value':
            critical_value = stats.t.ppf(1 - input_value / 2, n_data_points)
        else:
            critical_value = input_value

        # Estimate Gaussian noise
        sorted_wavelet = np.sort(np.abs(np.diff(data) / 1.4))
        sigma_noise = sorted_wavelet[int(round(0.682 * (n_data_points - 1)))]

        is_change_point = np.zeros(n_data_points, dtype=bool)
        is_change_point[0] = is_change_point[-1] = True

        first_change_point = 0
        current_change_point = 0
        next_change_point = n_data_points - 1

        while current_change_point < n_data_points - 1:
            if next_change_point - current_change_point >= min_data_points:
                data_segment = data[current_change_point + 1:next_change_point + 1]
                T, CP = self.t_test_cp(data_segment, sigma_noise)

                if T > critical_value:
                    is_change_point[current_change_point + CP] = True
                    next_change_point = current_change_point + CP

                    if first_change_point == 0:
                        first_change_point = CP
                else:
                    current_change_point = next_change_point
                    if current_change_point < n_data_points - 1:
                        next_change_point = np.where(is_change_point[current_change_point + 1:])[0][0] + current_change_point + 1
            else:
                current_change_point = next_change_point
                if current_change_point < n_data_points - 1:
                    next_change_point = np.where(is_change_point[current_change_point + 1:])[0][0] + current_change_point + 1

        change_point_sequence = np.zeros(n_data_points)
        change_points = np.where(is_change_point)[0]
        for i in range(len(change_points) - 1):
            start, stop = change_points[i], change_points[i + 1]
            change_point_sequence[start:stop] = np.mean(data[start:stop])

        n_change_points = np.sum(is_change_point) - 2
        return change_point_sequence, n_change_points, first_change_point

    def t_test_cp(self, data_segment, sigma_noise):
        n_data_points = len(data_segment)
        T, CP = 0, 0

        cum_sum = np.cumsum(data_segment)
        total_sum = cum_sum[-1]

        for n in range(2, n_data_points - 1):
            mu1 = cum_sum[n] / n
            mu2 = (total_sum - cum_sum[n]) / (n_data_points - n)
            t = abs(mu2 - mu1) / sigma_noise / np.sqrt(1/n + 1/(n_data_points - n))

            if t > T:
                T, CP = t, n

        return T, CP

    def kmeans_elkan(self, data, initial_centers):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=len(initial_centers), init=initial_centers.reshape(-1, 1), n_init=1, algorithm='elkan')
        labels = kmeans.fit_predict(data.reshape(-1, 1))
        return kmeans.cluster_centers_.flatten(), labels

    def agg_cluster(self, data, data_fit):
        n_clusters = len(np.unique(data_fit))
        all_data_fits = np.zeros((len(data), n_clusters))

        for k in range(n_clusters, 0, -1):
            if k == n_clusters:
                all_data_fits[:, k-1] = data_fit
            else:
                clustering = AgglomerativeClustering(n_clusters=k)
                labels = clustering.fit_predict(data_fit.reshape(-1, 1))
                centers = np.array([np.mean(data[labels == i]) for i in range(k)])
                all_data_fits[:, k-1] = centers[labels]

        return all_data_fits

    def compute_ic(self, data, all_data_fits, ic_type='BIC_GMM'):
        n_sequences = all_data_fits.shape[1]
        metrics = np.zeros(n_sequences)

        for k in range(n_sequences):
            if ic_type in ['BIC_GMM', 'AIC_GMM']:
                gmm = GaussianMixture(n_components=k+1, random_state=0)
                gmm.fit(data.reshape(-1, 1))
                if ic_type == 'BIC_GMM':
                    metrics[k] = gmm.bic(data.reshape(-1, 1))
                else:
                    metrics[k] = gmm.aic(data.reshape(-1, 1))
            elif ic_type == 'BIC_RSS':
                n_data_points = len(data)
                n_change_points = np.sum(np.diff(all_data_fits[:, k]) != 0)
                n_states = k + 1
                rss = np.sum((data - all_data_fits[:, k])**2)
                metrics[k] = n_data_points * np.log(rss / n_data_points) + (n_change_points + n_states) * np.log(n_data_points)

        # Add a small penalty for fewer states to encourage more states
        metrics += np.arange(n_sequences)[::-1] * 0.1

        n_states = np.argmin(metrics) + 1

        # Handle potential divide by zero
        metric_range = np.max(metrics) - np.min(metrics)
        if metric_range > 0:
            metrics = (metrics - np.min(metrics)) / metric_range
        else:
            metrics = np.zeros_like(metrics)

        return metrics, n_states

    def find_events(self, data_fit, threshold=0.01, min_event_duration=5):
        unique_states = np.unique(data_fit)
        if len(unique_states) < 2:
            return np.array([])

        state_changes = np.where(np.diff(data_fit) != 0)[0]

        if len(state_changes) == 0:
            return np.array([])

        all_events = []
        current_start = 0
        current_state = data_fit[0]

        for i, change in enumerate(state_changes):
            if change - current_start >= min_event_duration:
                all_events.append([current_start, change, current_state, data_fit[change]])
            current_start = change
            current_state = data_fit[change]

        # Add the last event
        if len(data_fit) - current_start >= min_event_duration:
            all_events.append([current_start, len(data_fit), current_state, data_fit[-1]])

        if not all_events:
            return np.array([])

        all_events = np.array(all_events)

        # Filter events based on threshold
        data_range = np.max(data_fit) - np.min(data_fit)
        normalized_threshold = threshold * data_range
        significant_events = np.abs(all_events[:, 2] - all_events[:, 3]) > normalized_threshold

        return all_events[significant_events]

    def plot_results(self, time_series, data_fit, best_fit, events, roi_index):
        self.plot_widget.clear()

        # Time series plot
        plot_item = self.plot_widget.addPlot(row=0, col=0)
        plot_item.plot(np.arange(len(time_series)), time_series, pen='b')
        if best_fit is not None:
            plot_item.plot(np.arange(len(best_fit)), best_fit, pen='r')
        plot_item.setTitle(f"ROI {roi_index+1} Time Series")
        plot_item.setLabel('bottom', 'Frame')
        plot_item.setLabel('left', 'Intensity')

        # Histogram plot
        hist_item = self.plot_widget.addPlot(row=0, col=1)
        y, x = np.histogram(time_series, bins=50)
        hist_item.plot(x, y, stepMode=True, fillLevel=0, brush='b')
        hist_item.setTitle(f"ROI {roi_index+1} Histogram")
        hist_item.setLabel('bottom', 'Intensity')
        hist_item.setLabel('left', 'Count')

        # Information Criterion plot
        ic_item = self.plot_widget.addPlot(row=0, col=2)
        if data_fit is not None and best_fit is not None:
            ic_type = self.ic_combo.currentText()
            metrics, _ = self.compute_ic(time_series, np.column_stack([data_fit, best_fit]), ic_type)
            ic_item.plot(range(1, len(metrics)+1), metrics, pen='b')
            ic_item.setTitle(f"ROI {roi_index+1} {ic_type}")
            ic_item.setLabel('bottom', 'Number of States')
            ic_item.setLabel('left', 'IC Value')
        else:
            ic_item.setTitle(f"ROI {roi_index+1} IC (Not Available)")

        # Add event boundaries if events are provided
        if events is not None and len(events) > 0:
            for event in events:
                plot_item.addItem(pg.InfiniteLine(event[0], angle=90, pen='y'))

        self.plot_widget.show()

    def get_dwell_times(self):
        selected_items = self.roi_list.selectedItems()
        selected_rois = [self.roi_list.row(item) for item in selected_items]
        if not selected_rois:
            print("No ROIs selected")
            return

        all_durations = []
        for roi_index in selected_rois:
            roi = self.rois[roi_index]
            time_series = roi.getArrayRegion(self.microview.window_manager.current_window.image,
                                             self.microview.window_manager.current_window.get_image_item()).mean(axis=(1,2))

            # Perform analysis to get best_fit
            sigma_noise = self.estimate_noise(time_series)
            data_fit, _ = self.div_segment(time_series, sigma_noise)
            all_data_fits = self.agg_cluster(time_series, data_fit)
            _, n_states = self.compute_ic(time_series, all_data_fits, self.ic_combo.currentText())
            best_fit = all_data_fits[:, n_states-1]

            events = self.find_events(best_fit)
            durations = events[:, 2]
            all_durations.append(durations)

        # Plot dwell time histograms
        self.plot_widget.clear()
        n_states = max(len(np.unique(d)) for d in all_durations)
        for i in range(n_states):
            hist_item = self.plot_widget.addPlot(row=i//3, col=i%3)
            for j, durations in enumerate(all_durations):
                state_durations = durations[events[:, 3] == i+1]
                y, x = np.histogram(state_durations, bins=30)
                hist_item.plot(x, y, stepMode=True, fillLevel=0, brush=self.channel_colors[j % len(self.channel_colors)])
            hist_item.setTitle(f"State {i+1} Dwell Times")
            hist_item.setLogMode(x=True, y=False)

    def get_state_occupancy(self):
        selected_items = self.roi_list.selectedItems()
        selected_rois = [self.roi_list.row(item) for item in selected_items]
        if not selected_rois:
            print("No ROIs selected")
            return

        all_occupancies = []
        for roi_index in selected_rois:
            roi = self.rois[roi_index]
            time_series = roi.getArrayRegion(self.microview.window_manager.current_window.image,
                                             self.microview.window_manager.current_window.get_image_item()).mean(axis=(1,2))

            # Perform analysis to get best_fit
            sigma_noise = self.estimate_noise(time_series)
            data_fit, _ = self.div_segment(time_series, sigma_noise)
            all_data_fits = self.agg_cluster(time_series, data_fit)
            _, n_states = self.compute_ic(time_series, all_data_fits, self.ic_combo.currentText())
            best_fit = all_data_fits[:, n_states-1]

            unique_states, counts = np.unique(best_fit, return_counts=True)
            occupancy = counts / len(best_fit)
            all_occupancies.append(occupancy)

        # Plot state occupancy
        self.plot_widget.clear()
        plot_item = self.plot_widget.addPlot(title="State Occupancy")
        bar_graph = pg.BarGraphItem(x=range(len(all_occupancies[0])), height=np.mean(all_occupancies, axis=0),
                                    width=0.6, brush='b')
        plot_item.addItem(bar_graph)
        plot_item.setLabel('left', 'Occupancy')
        plot_item.setLabel('bottom', 'State')



# Add this line at the end of the file
Plugin = FluorescenceAnalysisPlugin
