from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="microview",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A powerful and flexible image analysis tool for microscopy data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/microview",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "PyQt5",
        "pyqtgraph",
        "scikit-image",
        "tifffile",
        "nd2reader",
    ],
    extras_require={
        "dev": ["pytest", "flake8", "black"],
    },
    entry_points={
        "console_scripts": [
            "microview=microview.main:main",
        ],
    },
)
