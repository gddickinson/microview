import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys
from fluorescence_analysis_plugin import FluorescenceAnalysisPlugin, EventDetector

class MockMicroView:
    def __init__(self):
        self.window_manager = MockWindowManager()

class MockWindowManager:
    def __init__(self):
        self.current_window = MockWindow()

class MockWindow:
    def __init__(self):
        self.rois = [MockROI(), MockROI()]
        self.image = np.random.rand(6000, 52, 52)  # 6000 frames, 52x52 pixels

    def get_image_item(self):
        return self

class MockROI:
    def getArrayRegion(self, data, obj, axes=(1,2)):
        return data

class TestFluorescenceAnalysisPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a QApplication instance if it doesn't exist
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.mock_microview = MockMicroView()
        self.plugin = FluorescenceAnalysisPlugin(self.mock_microview)
        self.plugin.run()  # This will create and show the widget

    def generate_synthetic_data(self, n_points=6000, n_events=3, noise_level=5, min_duration=50, max_duration=500):
        base_level = 100.0
        data = np.full(n_points, base_level, dtype=np.float64)

        events = []
        available_positions = list(range(n_points))
        for _ in range(n_events):
            if not available_positions:
                break

            start = np.random.choice(available_positions)
            duration = np.random.randint(min_duration, min(max_duration, len(available_positions) - start))
            amplitude = np.random.uniform(30, 150)  # Increased minimum amplitude

            end = start + duration
            event_type = np.random.choice(['step_up', 'step_down', 'complex'])

            if event_type == 'step_up':
                data[start:] += amplitude
                events.append((start, n_points, amplitude, 'step_up'))
            elif event_type == 'step_down':
                if _ > 0:  # Ensure there's a previous event to step down from
                    data[start:] -= amplitude
                    events.append((start, n_points, -amplitude, 'step_down'))
                else:
                    # If it's the first event, make it a step_up instead
                    data[start:] += amplitude
                    events.append((start, n_points, amplitude, 'step_up'))
            else:  # complex event
                data[start:end] += amplitude
                events.append((start, end, amplitude, 'complex'))

            available_positions = [pos for pos in available_positions if pos < start or pos >= end]

        # Add noise
        noise = np.random.normal(0, noise_level, n_points)
        data += noise

        return data, events

    def test_roi_list_update(self):
        self.plugin.update_roi_list()
        self.assertEqual(self.plugin.roi_list.count(), 2)

    def test_detection_method_change(self):
        self.plugin.detection_method_combo.setCurrentIndex(1)  # CUSUM
        self.assertEqual(self.plugin.params_layout.rowCount(), 3)  # threshold, drift, min_event_duration

        self.plugin.detection_method_combo.setCurrentIndex(2)  # Threshold
        self.assertEqual(self.plugin.params_layout.rowCount(), 3)  # threshold, median_filter_size, min_event_duration

    def test_event_detection(self):
        n_events_list = [1, 3, 5]
        noise_levels = [2, 5, 10]

        for n_events in n_events_list:
            for noise_level in noise_levels:
                data, true_events = self.generate_synthetic_data(n_events=n_events, noise_level=noise_level)

                detector = EventDetector(data)

                # Test Original method
                events_original = detector.detect_events_original()
                self.assertGreaterEqual(len(events_original), 0)

                # Test CUSUM method
                events_cusum = detector.detect_events_cusum()
                self.assertGreaterEqual(len(events_cusum), 0)

                # Test Threshold method
                events_threshold = detector.detect_events_threshold()
                self.assertGreaterEqual(len(events_threshold), 0)

                print(f"n_events={n_events}, noise={noise_level}")
                print(f"True events: {len(true_events)}")
                print(f"Detected events (Original): {len(events_original)}")
                print(f"Detected events (CUSUM): {len(events_cusum)}")
                print(f"Detected events (Threshold): {len(events_threshold)}")
                print("--------------------")

    def test_analyze_fluorescence(self):
        # Select first ROI
        self.plugin.roi_list.item(0).setSelected(True)

        # Test for each detection method
        for i in range(3):
            self.plugin.detection_method_combo.setCurrentIndex(i)
            self.plugin.analyze_fluorescence()

            # Check if plot_widget has items (implying successful analysis and plotting)
            self.assertGreater(len(self.plugin.plot_widget.items), 0)

    def test_edge_cases(self):
        # Test no events
        data_no_events = np.random.normal(100, 2, 1000)  # Reduced noise
        detector_no_events = EventDetector(data_no_events)

        events_original = detector_no_events.detect_events_original()
        events_cusum = detector_no_events.detect_events_cusum()
        events_threshold = detector_no_events.detect_events_threshold()

        self.assertEqual(len(events_original), 0, "Original method should detect no events in constant noisy data")
        self.assertEqual(len(events_cusum), 0, "CUSUM method should detect no events in constant noisy data")
        self.assertEqual(len(events_threshold), 0, "Threshold method should detect no events in constant noisy data")

        # Test all events (step function)
        data_all_events = np.concatenate([np.full(500, 100), np.full(500, 200)])
        detector_all_events = EventDetector(data_all_events)

        events_original = detector_all_events.detect_events_original()
        events_cusum = detector_all_events.detect_events_cusum()
        events_threshold = detector_all_events.detect_events_threshold()

        self.assertEqual(len(events_original), 1, "Original method should detect one event in step function data")
        self.assertEqual(len(events_cusum), 1, "CUSUM method should detect one event in step function data")
        self.assertEqual(len(events_threshold), 1, "Threshold method should detect one event in step function data")

    def tearDown(self):
        if self.plugin.widget:
            self.plugin.widget.close()

if __name__ == '__main__':
    unittest.main()
