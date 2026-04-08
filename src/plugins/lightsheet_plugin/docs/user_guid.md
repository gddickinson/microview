# LightSheet Microscopy Viewer User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Main Interface](#main-interface)
4. [Data Generation](#data-generation)
5. [Data Visualization](#data-visualization)
6. [Raw Data Viewer](#raw-data-viewer)
7. [Blob Detection and Analysis](#blob-detection-and-analysis)
8. [File Operations](#file-operations)
9. [Troubleshooting](#troubleshooting)

## Introduction

The LightSheet Microscopy Viewer is a powerful tool for visualizing, analyzing, and simulating lightsheet microscopy data. This guide will walk you through the main features and functionalities of the application.

## Getting Started

After installing the application, launch it by running:

python biologicalSimulator/lightsheetviewer.py

## Main Interface

The main interface consists of several docked widgets:

- **XY, XZ, and YZ Views**: 2D slice views of the data.
- **3D View**: Interactive 3D rendering of the data.
- **Data Generation**: Controls for generating synthetic data.
- **Visualization Control**: Options for adjusting the 3D visualization.
- **Playback Control**: Controls for navigating through time series data.
- **Blob Detection**: Tools for detecting and analyzing blobs in the data.

## Data Generation

1. Navigate to the "Data Generation" dock.
2. Set the desired parameters:
   - Number of Volumes
   - Number of Blobs
   - Noise Level
   - Movement Speed
3. Check "Generate Structured Data" if you want to confine blobs to specific regions.
4. Click "Generate New Data" to create synthetic data.

## Data Visualization

1. Use the sliders in the XY, XZ, and YZ views to navigate through the data.
2. In the 3D View:
   - Left-click and drag to rotate.
   - Right-click and drag to zoom.
   - Middle-click and drag to pan.
3. Adjust visualization settings in the "Visualization Control" dock:
   - Toggle channel visibility and adjust opacity.
   - Change the rendering mode (Points, Surface, Wireframe).
   - Adjust the threshold for surface rendering.
   - Enable/disable downsampling for large datasets.

## Raw Data Viewer

1. Go to "Raw Data" in the menu bar and select "Show Raw Data Viewer".
2. Use the Volume slider to navigate through different time points.
3. Use the Channel dropdown to select individual channels or view all channels overlaid.

## Blob Detection and Analysis

1. Navigate to the "Blob Detection" dock.
2. Set the detection parameters:
   - Max Sigma
   - Num Sigma
   - Threshold
3. Click "Detect Blobs" to run the detection algorithm.
4. Use "Show Blob Results" to view detailed information about detected blobs.
5. Click "Analyze Blobs" for additional statistical analysis.

## File Operations

- To save data: File -> Save Data
- To load data: File -> Load Data
- To import microscope data: File -> Import Microscope Data

## Troubleshooting

- If the application crashes, check the console for error messages.
- For large datasets, try enabling downsampling to improve performance.
- If blob detection is not working as expected, try adjusting the threshold and sigma values.

For additional help or to report issues, please refer to the project's GitHub repository.
