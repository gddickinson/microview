import numpy as np
from skimage import filters, measure
from PyQt5.QtWidgets import QMessageBox
from particle_analysis import ParticleAnalysisResults
from scipy import stats
import pyqtgraph as pg

class AnalysisOperations:
    def __init__(self, parent):
        self.parent = parent

    def measure(self):
        if self.parent.window_management.current_window:
            image = self.parent.window_management.current_window.image
            print(f"Mean: {np.mean(image)}")
            print(f"Std Dev: {np.std(image)}")
            print(f"Min: {np.min(image)}")
            print(f"Max: {np.max(image)}")

    def findMaxima(self):
        if self.parent.window_management.current_window:
            image = self.parent.window_management.current_window.image
            local_max = filters.peak_local_max(image)
            print(f"Found {len(local_max)} local maxima")

    def run_particle_analysis(self):
        if self.parent.window_management.current_window is None:
            QMessageBox.warning(self.parent, "No Image", "Please open an image first.")
            return

        try:
            image = self.parent.window_management.current_window.image
            analysis_dialog = ParticleAnalysisResults(self.parent, image)
            analysis_dialog.analysisComplete.connect(self.parent.on_particle_analysis_complete)
            analysis_dialog.exec_()

        except Exception as e:
            print(f"Error in particle analysis: {str(e)}")
            QMessageBox.critical(self.parent, "Error", f"Error in particle analysis: {str(e)}")

    def colocalization_analysis(self):
        if self.parent.window_management.current_window and hasattr(self.parent.window_management.current_window, 'rois') and len(self.parent.window_management.current_window.rois) == 1:
            roi = self.parent.window_management.current_window.rois[0]
            image_view = self.parent.window_management.current_window
            image = image_view.getImageItem().image

            if image.ndim != 3 or image.shape[2] != 2:
                print("Colocalization analysis requires a two-channel image")
                return

            roi_data = roi.getArrayRegion(image, image_view.getImageItem())
            channel1 = roi_data[:, :, 0].flatten()
            channel2 = roi_data[:, :, 1].flatten()

            pearson_corr, _ = stats.pearsonr(channel1, channel2)
            manders_m1 = np.sum(channel1[channel2 > 0]) / np.sum(channel1)
            manders_m2 = np.sum(channel2[channel1 > 0]) / np.sum(channel2)

            print("Colocalization Analysis:")
            print(f"Pearson's correlation coefficient: {pearson_corr:.3f}")
            print(f"Manders' coefficient M1: {manders_m1:.3f}")
            print(f"Manders' coefficient M2: {manders_m2:.3f}")

            # Create a scatter plot of the two channels
            scatter_plot = pg.plot(title="Channel Intensity Scatter Plot")
            scatter_plot.plot(channel1, channel2, pen=None, symbol='o', symbolSize=5, symbolPen=None, symbolBrush=(100, 100, 255, 50))
            scatter_plot.setLabel('left', 'Channel 2 Intensity')
            scatter_plot.setLabel('bottom', 'Channel 1 Intensity')
        else:
            print("Please select a single ROI for colocalization analysis")

    def open_analysis_console(self):
        if self.parent.window_management.current_window:
            image = self.parent.window_management.current_window.image
            self.parent.analysis_console = self.parent.scikit_analysis_console(image)
            self.parent.analysis_console.analysisCompleted.connect(self.parent.display_analysis_result)
            self.parent.analysis_console.show()
