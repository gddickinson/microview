import numpy as np
from PyQt5.QtWidgets import QInputDialog, QMessageBox

class MathOperations:
    def __init__(self, parent):
        self.parent = parent

    def mathOperation(self, operation):
        if self.parent.window_manager.current_window:
            value, ok = QInputDialog.getDouble(self.parent, "Input", "Enter value:")
            if ok:
                image = self.parent.window_manager.current_window.image
                original_dtype = image.dtype

                # Convert image to float64 for calculations
                image = image.astype(np.float64)

                if operation == 'add':
                    result = image + value
                elif operation == 'subtract':
                    result = image - value
                elif operation == 'multiply':
                    result = image * value
                elif operation == 'divide':
                    # Avoid division by zero
                    if value == 0:
                        QMessageBox.warning(self.parent, "Error", "Cannot divide by zero.")
                        return
                    result = image / value
                else:
                    QMessageBox.warning(self.parent, "Error", f"Unknown operation: {operation}")
                    return

                # Clip the result to the range of the original dtype
                info = np.iinfo(original_dtype)
                result = np.clip(result, info.min, info.max)

                # Convert back to the original dtype
                result = result.astype(original_dtype)

                self.parent.window_manager.current_window.setImage(result)
                print(f"Applied {operation} operation with value {value}")
        else:
            QMessageBox.warning(self.parent, "Error", "No image window is currently active.")
