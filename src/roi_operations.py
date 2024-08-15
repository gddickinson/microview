from PyQt5.QtWidgets import QFileDialog
import json
from roi import RectROI, EllipseROI, LineROI

class ROIOperations:
    def __init__(self, parent):
        self.parent = parent

    def addROI(self, roi_type):
        if self.parent.window_manager.current_window:
            try:
                image_window = self.parent.window_manager.current_window
                roi = image_window.add_roi(roi_type)
                print(f"ROI added to view: {roi}")
            except Exception as e:
                print(f"Error adding ROI: {str(e)}")

    def removeAllROIs(self):
        if self.parent.window_manager.current_window:
            try:
                image_window = self.parent.window_manager.current_window
                for roi in image_window.rois[:]:
                    image_window.remove_roi(roi)
                print("All ROIs removed")
            except Exception as e:
                print(f"Error removing all ROIs: {str(e)}")

    def save_rois_dialog(self):
        filename, _ = QFileDialog.getSaveFileName(self.parent, "Save ROIs", "", "JSON Files (*.json)")
        if filename:
            self.save_rois(filename)

    def load_rois_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self.parent, "Load ROIs", "", "JSON Files (*.json)")
        if filename:
            self.open_rois(filename)

    def save_rois(self, filename):
        try:
            current_window = self.parent.window_manager.current_window
            if current_window is None or not hasattr(current_window, 'rois'):
                print("No ROIs to save.")
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

            print(f"Saved {len(roi_data)} ROIs to {filename}")
        except Exception as e:
            print(f"Error saving ROIs to {filename}: {str(e)}")

    def open_rois(self, filename):
        try:
            with open(filename, 'r') as f:
                roi_data = json.load(f)

            rois = []
            current_window = self.parent.window_manager.current_window

            if current_window is None:
                print("No current window to add ROIs to.")
                return rois

            for roi_info in roi_data:
                roi_type = roi_info['type']
                pos = roi_info['pos']
                size = roi_info['size']

                roi = current_window.add_roi(roi_type)
                roi.setPos(pos)
                if hasattr(roi, 'setSize'):
                    roi.setSize(size)
                elif isinstance(roi, LineROI):
                    roi.setPoints([pos, [pos[0] + size[0], pos[1] + size[1]]])

                rois.append(roi)

            print(f"Loaded {len(rois)} ROIs from {filename}")
            return rois
        except Exception as e:
            print(f"Error loading ROIs from {filename}: {str(e)}")
            return []
