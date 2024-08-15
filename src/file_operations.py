from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os
import json
import logging
from image_window import ImageWindow

logger = logging.getLogger(__name__)

class FileOperations:
    def __init__(self, parent):
        self.parent = parent

    def open_file(self):
        file_types = "All Supported Files ("
        file_types += " ".join(f"*{ext}" for ext in self.parent.file_loader.supported_extensions)
        file_types += ");;"
        file_types += ";;".join([f"{ext.upper()[1:]} Files (*{ext})" for ext in self.parent.file_loader.supported_extensions])

        fileName, _ = QFileDialog.getOpenFileName(self.parent, "Open Image", "", file_types)
        if fileName:
            self.load_image(fileName)

    def load_image(self, fileName):
        try:
            result = self.parent.file_loader.load_file(fileName)
            if result is not None:
                image, metadata = result

                metadata = metadata or {}

                try:
                    self.parent.file_loader.save_metadata(fileName, metadata)
                except Exception as e:
                    logger.warning(f"Failed to save metadata: {str(e)}")

                window = ImageWindow(image, os.path.basename(fileName), metadata)
                self.parent.window_management.add_window(window)

                window.timeChanged.connect(self.parent.on_time_slider_changed)

                self.parent.variable_management.push_variables({
                    'current_image': window.get_image(),
                    'current_metadata': window.get_metadata(),
                    'current_window': window
                })

                logger.info(f"Loaded image: {fileName}")

                self.update_recent_files(fileName)

        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            QMessageBox.critical(self.parent, "Error", f"Failed to load image: {str(e)}")


    def update_recent_files(self, fileName):
        if fileName in self.parent.recent_files:
            self.parent.recent_files.remove(fileName)
        self.parent.recent_files.insert(0, fileName)
        self.parent.recent_files = self.parent.recent_files[:10]  # Keep only 10 most recent
        self.parent.update_recent_files_menu()
        self.parent.config_manager.save_config()

    def save_file(self):
        if self.parent.window_management.current_window:
            fileName, _ = QFileDialog.getSaveFileName(self.parent, "Save Image", "", "TIFF Files (*.tiff)")
            if fileName:
                import tifffile
                tifffile.imwrite(fileName, self.parent.window_management.current_window.get_image())

    def save_rois(self, filename):
        try:
            current_window = self.parent.window_manager.current_window
            if current_window is None or not hasattr(current_window, 'rois'):
                logger.warning("No ROIs to save.")
                return

            roi_data = []
            for roi in current_window.rois:
                roi_info = {
                    'type': roi.__class__.__name__.lower().replace('roi', ''),
                    'pos': roi.pos().tolist() if hasattr(roi, 'pos') else roi.getState()['pos'],
                    'size': roi.size().tolist() if hasattr(roi, 'size') else roi.getState()['size'],
                }
                roi_data.append(roi_info)

            with open(filename, 'w') as f:
                json.dump(roi_data, f)

            logger.info(f"Saved {len(roi_data)} ROIs to {filename}")
        except Exception as e:
            logger.error(f"Error saving ROIs to {filename}: {str(e)}")

    def load_rois(self, filename):
        try:
            with open(filename, 'r') as f:
                roi_data = json.load(f)

            rois = []
            current_window = self.parent.window_manager.current_window

            if current_window is None:
                logger.warning("No current window to add ROIs to.")
                return rois

            for roi_info in roi_data:
                roi_type = roi_info['type']
                pos = roi_info['pos']
                size = roi_info['size']

                roi = current_window.add_roi(roi_type)
                roi.setPos(pos)
                if hasattr(roi, 'setSize'):
                    roi.setSize(size)
                elif isinstance(roi, self.parent.LineROI):
                    roi.setPoints([pos, [pos[0] + size[0], pos[1] + size[1]]])

                rois.append(roi)

            logger.info(f"Loaded {len(rois)} ROIs from {filename}")
            return rois
        except Exception as e:
            logger.error(f"Error loading ROIs from {filename}: {str(e)}")
            return []

    def save_rois_dialog(self):
        filename, _ = QFileDialog.getSaveFileName(self.parent, "Save ROIs", "", "JSON Files (*.json)")
        if filename:
            self.save_rois(filename)

    def load_rois_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self.parent, "Load ROIs", "", "JSON Files (*.json)")
        if filename:
            self.load_rois(filename)
