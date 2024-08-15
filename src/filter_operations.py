from PyQt5.QtWidgets import QInputDialog

class FilterOperations:
    def __init__(self, parent):
        self.parent = parent

    def gaussianBlur(self):
        if self.parent.window_manager.current_window:
            sigma, ok = QInputDialog.getDouble(self.parent, "Gaussian Blur", "Enter sigma value:")
            if ok:
                image = self.parent.window_manager.current_window.image
                blurred = self.parent.image_processor.gaussian_blur(image, sigma)
                self.parent.window_manager.current_window.setImage(blurred)

    def medianFilter(self):
        if self.parent.window_manager.current_window:
            size, ok = QInputDialog.getInt(self.parent, "Median Filter", "Enter filter size:")
            if ok:
                image = self.parent.window_manager.current_window.image
                filtered = self.parent.image_processor.median_filter(image, size)
                self.parent.window_manager.current_window.setImage(filtered)

    def sobelEdge(self):
        if self.parent.window_manager.current_window:
            image = self.parent.window_manager.current_window.image
            edges = self.parent.image_processor.sobel_edge(image)
            self.parent.window_manager.current_window.setImage(edges)
