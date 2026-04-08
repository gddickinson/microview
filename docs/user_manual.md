MicroView User Manual

Table of Contents

1. Introduction
2. Installation
3. Getting Started
4. Main Interface
5. Loading and Managing Images
6. Image Visualization
7. Region of Interest (ROI) Analysis
8. Image Processing
9. Particle Analysis
10. Plugins
11. Exporting Results
12. Troubleshooting
13. FAQ

1. Introduction

MicroView is a powerful and flexible image analysis tool designed for microscopy data. It provides a user-friendly interface for viewing, processing, and analyzing multi-dimensional image data.

2. Installation

a) Ensure you have Python 3.7 or later installed on your system.
b) Clone the MicroView repository:
   git clone https://github.com/yourusername/microview.git
c) Navigate to the MicroView directory:
   cd microview
d) Install the required dependencies:
   pip install -r requirements.txt

3. Getting Started

To launch MicroView:

a) Open a terminal or command prompt.
b) Navigate to the MicroView directory.
c) Run the following command:
   python main.py

4. Main Interface

The MicroView interface consists of:

- Menu Bar: File, Edit, View, ROI, Analysis, Plugins, Help
- Toolbar: Quick access to common functions
- Image Display Area: Central area where images are displayed
- Info Panel: Displays metadata and current image information
- ROI Panel: Manages Regions of Interest
- Status Bar: Displays current status and cursor information

5. Loading and Managing Images

To load an image:
a) Click "File" > "Open" in the menu bar.
b) Select your image file (supported formats: TIFF, ND2, CZI, etc.).
c) The image will open in the main display area.

To manage multiple images:
- Use the Window menu to switch between open images.
- Use "File" > "Close" to close the current image.

6. Image Visualization

- Zoom: Use the mouse wheel or the zoom tools in the toolbar.
- Pan: Click and drag the image with the middle mouse button.
- Adjust Brightness/Contrast: Use the histogram tool in the toolbar.
- For multi-dimensional images:
  - Use the time slider to navigate through time points.
  - Use the z-stack slider to navigate through z-slices.
  - Use the channel selector to switch between or combine channels.

7. Region of Interest (ROI) Analysis

To add an ROI:
a) Click "ROI" in the menu bar.
b) Select the ROI type (Rectangle, Ellipse, Polygon, etc.).
c) Draw the ROI on the image.

To analyze an ROI:
a) Select the ROI.
b) Click "Analyze" in the ROI panel.
c) View results in the Info Panel.

8. Image Processing

MicroView offers various image processing tools:

- Filters: Gaussian, Median, etc.
- Segmentation: Thresholding, Watershed, etc.
- Morphological operations: Erosion, Dilation, etc.

To apply processing:
a) Select the desired tool from the "Process" menu.
b) Adjust parameters in the dialog that appears.
c) Click "Apply" to process the image.

9. Particle Analysis

To perform particle analysis:
a) Ensure your image is properly segmented.
b) Click "Analysis" > "Particle Analysis" in the menu bar.
c) Adjust analysis parameters in the dialog.
d) Click "Run" to perform the analysis.
e) View results in the Results Table.

10. Plugins

MicroView supports plugins to extend its functionality:

a) To install a plugin, place the plugin file in the "plugins" directory.
b) Restart MicroView.
c) Access the plugin from the "Plugins" menu.

11. Exporting Results

To export analysis results:
a) Click "File" > "Export" in the menu bar.
b) Choose the export format (CSV, Excel, etc.).
c) Select the destination and click "Save".

To export processed images:
a) Click "File" > "Save As" in the menu bar.
b) Choose the image format.
c) Select the destination and click "Save".

12. Troubleshooting

If you encounter issues:
a) Check the console for error messages.
b) Ensure all dependencies are correctly installed.
c) Verify that your input files are not corrupted.
d) Restart MicroView and try again.

If problems persist, please report the issue on our GitHub page.

13. FAQ

Q: What image formats are supported?
A: MicroView supports TIFF, ND2, CZI, and various other microscopy formats.

Q: Can I use MicroView for batch processing?
A: Yes, you can use the batch processing feature under "File" > "Batch Process".

Q: How do I create custom plugins?
A: Refer to the Developer Guide in the docs folder for information on creating plugins.

For more questions, please check our GitHub issues page or contact support.
