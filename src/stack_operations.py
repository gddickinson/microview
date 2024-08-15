from PyQt5.QtWidgets import QMessageBox

class StackOperations:
    def __init__(self, parent):
        self.parent = parent

    def zProjectMax(self):
        if self.parent.window_manager.current_window:
            image = self.parent.window_manager.current_window.image
            if image.ndim == 3:
                try:
                    projected = self.parent.image_processor.z_project(image, method='max')
                    self.parent.window_manager.current_window.setImage(projected)
                except Exception as e:
                    self.parent.logger.error(f"Error in zProjectMax: {str(e)}")
                    QMessageBox.critical(self.parent, "Error", f"Error in maximum intensity projection: {str(e)}")

    def zProjectMean(self):
        if self.parent.window_manager.current_window:
            image = self.parent.window_manager.current_window.image
            if image.ndim == 3:
                try:
                    projected = self.parent.image_processor.z_project(image, method='mean')
                    self.parent.window_manager.current_window.setImage(projected)
                except Exception as e:
                    self.parent.logger.error(f"Error in zProjectMean: {str(e)}")
                    QMessageBox.critical(self.parent, "Error", f"Error in mean intensity projection: {str(e)}")
