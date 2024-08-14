#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:59:20 2024

@author: george
"""

# filters.py

from typing import Callable, Any
import numpy as np
from skimage import filters, morphology, exposure, segmentation, restoration
from PyQt5.QtWidgets import QInputDialog, QMessageBox

class Filters:
    def __init__(self, parent):
        self.parent = parent

    def apply_filter(self, filter_func: Callable, *args: Any, **kwargs: Any) -> None:
        """
        Apply a filter function to the current image.

        Args:
            filter_func (Callable): The filter function to apply.
            *args: Positional arguments for the filter function.
            **kwargs: Keyword arguments for the filter function.

        Raises:
            ValueError: If no image is open.
        """
        if self.parent.current_window is None:
            raise ValueError("No image open. Please open an image first.")

        try:
            image = self.parent.current_window.image
            filtered = filter_func(image, *args, **kwargs)
            self.parent.loadImage(filtered)
        except Exception as e:
            raise RuntimeError(f"Error applying filter: {str(e)}")

    def sobel(self) -> None:
        """Apply Sobel edge detection filter."""
        self.apply_filter(filters.sobel)

    def canny(self) -> None:
        """Apply Canny edge detection filter."""
        sigma, ok = QInputDialog.getDouble(self.parent, "Canny Edge Detection", "Sigma:", 1.0, 0.1, 5.0)
        if ok:
            self.apply_filter(filters.canny, sigma=sigma)

    def gaussian(self) -> None:
        """Apply Gaussian smoothing filter."""
        sigma, ok = QInputDialog.getDouble(self.parent, "Gaussian Filter", "Sigma:", 1.0, 0.1, 5.0)
        if ok:
            self.apply_filter(filters.gaussian, sigma=sigma)

    def median(self) -> None:
        """Apply median filter."""
        size, ok = QInputDialog.getInt(self.parent, "Median Filter", "Size:", 3, 1, 11, 2)
        if ok:
            self.apply_filter(filters.median, size=size)

    def threshold_otsu(self) -> None:
        """Apply Otsu's thresholding."""
        self.apply_filter(lambda img: img > filters.threshold_otsu(img))

    def threshold_adaptive(self) -> None:
        """Apply adaptive thresholding."""
        block_size, ok = QInputDialog.getInt(self.parent, "Adaptive Threshold", "Block size:", 11, 3, 51, 2)
        if ok:
            self.apply_filter(filters.threshold_local, block_size=block_size)

    def erosion(self) -> None:
        """Apply erosion morphological operation."""
        size, ok = QInputDialog.getInt(self.parent, "Erosion", "Size:", 3, 1, 11, 2)
        if ok:
            self.apply_filter(morphology.erosion, morphology.square(size))

    def dilation(self) -> None:
        """Apply dilation morphological operation."""
        size, ok = QInputDialog.getInt(self.parent, "Dilation", "Size:", 3, 1, 11, 2)
        if ok:
            self.apply_filter(morphology.dilation, morphology.square(size))

    def contrast_stretching(self) -> None:
        """Apply contrast stretching."""
        self.apply_filter(exposure.rescale_intensity)

    def histogram_equalization(self) -> None:
        """Apply histogram equalization."""
        self.apply_filter(exposure.equalize_hist)

    def tv_denoise(self) -> None:
        """Apply total variation denoising."""
        weight, ok = QInputDialog.getDouble(self.parent, "TV Denoise", "Weight:", 0.1, 0.01, 1.0)
        if ok:
            self.apply_filter(restoration.denoise_tv_chambolle, weight=weight)

    def nl_means_denoising(self) -> None:
        """Apply non-local means denoising."""
        h, ok = QInputDialog.getDouble(self.parent, "Non-local Means Denoising", "h parameter:", 0.1, 0.01, 1.0)
        if ok:
            self.apply_filter(restoration.denoise_nl_means, h=h)

    def watershed(self) -> None:
        """Apply watershed segmentation."""
        self.apply_filter(segmentation.watershed)
