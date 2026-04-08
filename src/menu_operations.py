from PyQt5.QtWidgets import QAction

class MenuOperations:
    def __init__(self, parent):
        self.parent = parent

    def add_menu_item(self, menu_name, item_name, callback):
        if menu_name not in self.parent.menu_manager.menus:
            self.parent.menu_manager.menus[menu_name] = self.parent.menuBar().addMenu(menu_name)
        action = QAction(item_name, self.parent)
        action.triggered.connect(callback)
        self.parent.menu_manager.menus[menu_name].addAction(action)

    def show_user_guide(self):
        self.parent.QMessageBox.information(self.parent, "User Guide", "User guide content goes here.")

    def show_about(self):
        about_text = f"""
        MicroView version {self.parent.VERSION}

        A microscopy image viewer and analysis tool.

        Developed by [Your Name/Organization]
        """
        self.parent.QMessageBox.about(self.parent, "About MicroView", about_text)
