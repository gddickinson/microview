from setuptools import setup, find_packages

setup(
    name="lightsheet-viewer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "PyQt5",
        "pyqtgraph",
        "tifffile",
        "scikit-image",
    ],
    entry_points={
        "console_scripts": [
            "lightsheet-viewer=lightsheet_viewer.lightsheetviewer:main",
        ],
    },
)
