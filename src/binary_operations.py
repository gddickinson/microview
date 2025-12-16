class BinaryOperations:
    def __init__(self, parent):
        self.parent = parent

    def threshold(self):
        if self.parent.window_management.current_window:
            image = self.parent.window_management.current_window.image
            binary = self.parent.image_processor.threshold(image)
            self.parent.window_management.current_window.setImage(binary)

    def erode(self):
        if self.parent.window_management.current_window:
            image = self.parent.window_management.current_window.image
            eroded = self.parent.image_processor.erode(image)
            self.parent.window_management.current_window.setImage(eroded)

    def dilate(self):
        if self.parent.window_management.current_window:
            image = self.parent.window_management.current_window.image
            dilated = self.parent.image_processor.dilate(image)
            self.parent.window_management.current_window.setImage(dilated)
