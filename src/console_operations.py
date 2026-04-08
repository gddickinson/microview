from PyQt5.QtWidgets import QDockWidget
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

class MicroViewConsole(RichJupyterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._create_new_kernel()

    def _create_new_kernel(self):
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
        self.exit_requested.connect(stop)

    def push_variables(self, variables):
        self.kernel_manager.kernel.shell.push(variables)

class ConsoleOperations:
    def __init__(self, parent):
        self.parent = parent
        self.console = None
        self.console_dock = None

    def create_console_dock(self):
        if not self.parent.in_spyder:
            self.console = MicroViewConsole(self.parent)
            self.console_dock = QDockWidget("IPython Console", self.parent)
            self.console_dock.setWidget(self.console)
            self.console_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        else:
            self.console = None
            self.console_dock = None

    def toggle_console(self):
        if self.console_dock and self.console_dock.isVisible():
            self.console_dock.hide()
        elif self.console_dock:
            self.console_dock.show()

    def show_ipython_commands(self):
        commands = """
        Available IPython commands:
        - get_mv_var(name): Get a shared variable
        - ex: The MicroView instance
        - ex.loadImage(filename): Load an image file
        - ex.current_window: Access the current image window
        """
        self.parent.QMessageBox.information(self.parent, "IPython Commands", commands)
