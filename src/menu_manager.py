# menu_manager.py
from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QFileDialog
from PyQt5.QtCore import Qt

class MenuManager:
    def __init__(self, parent):
        self.parent = parent
        self.menus = {}
        self.recent_files_actions = []

    def create_menus(self):
        menubar = self.parent.menuBar()

        # File Menu
        self.menus['file'] = menubar.addMenu('File')
        self.add_action('file', 'Open', self.parent.openFile)
        self.add_action('file', 'Save', self.parent.saveFile)
        self.add_action('file', 'Close Current Window', self.parent.closeCurrentWindow)
        # Add Recent Files submenu
        self.menus['file'].addSeparator()  # Add a separator for clarity
        self.menus['recent_files'] = self.menus['file'].addMenu('Recent Files')
        self.update_recent_files_menu()

        # Edit Menu
        self.menus['edit'] = menubar.addMenu('Edit')
        self.add_action('edit', 'Undo', self.parent.undo)
        self.add_action('edit', 'Redo', self.parent.redo)

        # Process Menu
        self.menus['process'] = menubar.addMenu('Process')
        self.create_math_submenu()
        self.create_filters_submenu()
        self.create_binary_submenu()
        self.create_stack_submenu()
        # Add new items to the Process menu
        self.menus['process'].addSeparator()  # Add a separator for clarity
        self.add_action('process', 'Image Transformations', self.parent.open_transformations_dialog)
        self.add_action('process', 'Generate Synthetic Data', self.parent.open_synthetic_data_dialog)



        # Analyze Menu
        self.menus['analyze'] = menubar.addMenu('Analyze')
        self.add_action('analyze', 'Measure', self.parent.measure)
        self.add_action('analyze', 'Find Maxima', self.parent.findMaxima)
        self.add_action('analyze', 'Particle Analysis', self.parent.run_particle_analysis)
        self.add_action('analyze', 'Colocalization Analysis', self.parent.colocalization_analysis)
        scikit_menu = self.menus['analyze'].addMenu('Scikit-image Analysis')
        self.add_action(scikit_menu, 'Open Analysis Console', self.parent.open_analysis_console)
        # You can add more Scikit-image related actions here in the future


        # Plugins Menu
        self.menus['plugins'] = menubar.addMenu('Plugins')
        self.add_action('plugins', 'Load Plugin', self.parent.loadPlugin)
        self.add_action('plugins', 'Show Plugin Window', self.parent.togglePluginWindow)


        # Window Menu
        self.menus['window'] = menubar.addMenu('Window')
        self.add_action('window', 'Tile', self.parent.tileWindows)
        self.add_action('window', 'Cascade', self.parent.cascadeWindows)

        # View Menu
        self.create_view_menu()

        # ROI Menu
        self.menus['roi'] = menubar.addMenu('ROI')
        self.add_action('roi', 'Rectangle', lambda: self.parent.addROI('rectangle'))
        self.add_action('roi', 'Ellipse', lambda: self.parent.addROI('ellipse'))
        self.add_action('roi', 'Line', lambda: self.parent.addROI('line'))
        self.add_action('roi', 'Remove All ROIs', self.parent.removeAllROIs)
        self.add_action('roi', 'Save ROIs', self.parent.save_rois_dialog)
        self.add_action('roi', 'Load ROIs', self.parent.load_rois_dialog)
        self.add_action('roi', 'Toggle ROI Tools', self.parent.toggleROIToolsDock)


    def add_action(self, menu, action_name, slot):
        action = QAction(action_name, self.parent)
        action.triggered.connect(slot)
        if isinstance(menu, str):
            self.menus[menu].addAction(action)
        else:
            menu.addAction(action)

    def create_math_submenu(self):
        math_menu = self.menus['process'].addMenu('Math')
        self.add_action(math_menu, 'Add Constant', lambda: self.parent.mathOperation('add'))
        self.add_action(math_menu, 'Subtract Constant', lambda: self.parent.mathOperation('subtract'))
        self.add_action(math_menu, 'Multiply by Constant', lambda: self.parent.mathOperation('multiply'))
        self.add_action(math_menu, 'Divide by Constant', lambda: self.parent.mathOperation('divide'))

    def create_filters_submenu(self):
        filters_menu = self.menus['process'].addMenu('Filters')
        self.add_action(filters_menu, 'Gaussian Blur', self.parent.gaussianBlur)
        self.add_action(filters_menu, 'Median Filter', self.parent.medianFilter)
        self.add_action(filters_menu, 'Sobel Edge Detection', self.parent.sobelEdge)

    def create_binary_submenu(self):
        binary_menu = self.menus['process'].addMenu('Binary')
        self.add_action(binary_menu, 'Threshold', self.parent.threshold)
        self.add_action(binary_menu, 'Erode', self.parent.erode)
        self.add_action(binary_menu, 'Dilate', self.parent.dilate)

    def create_stack_submenu(self):
        stack_menu = self.menus['process'].addMenu('Stack')
        self.add_action(stack_menu, 'Z-Project (Max Intensity)', self.parent.zProjectMax)
        self.add_action(stack_menu, 'Z-Project (Mean Intensity)', self.parent.zProjectMean)

    def update_recent_files_menu(self):
        self.menus['recent_files'].clear()
        for file in self.parent.recent_files:
            action = QAction(file, self.parent)
            action.triggered.connect(lambda checked, f=file: self.parent.loadImage(f))
            self.menus['recent_files'].addAction(action)

    def create_view_menu(self):
        self.menus['view'] = self.parent.menuBar().addMenu('View')

        # For dock widgets, we directly add their toggleViewAction
        self.menus['view'].addAction(self.parent.info_dock.toggleViewAction())
        self.menus['view'].addAction(self.parent.z_profile_dock.toggleViewAction())

        # For custom toggle methods, we use add_action
        self.add_action('view', 'Toggle ROI Tools', self.parent.toggleROIToolsDock)
        self.add_action('view', 'Toggle Plugin Panel', self.parent.togglePluginDock)

        if self.parent.console_dock:
            self.menus['view'].addAction(self.parent.console_dock.toggleViewAction())
