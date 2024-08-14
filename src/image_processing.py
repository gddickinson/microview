#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:47:05 2024

@author: george
"""

# microview/image_processing.py


from typing import Union, Tuple
import numpy as np
from scipy import ndimage
from skimage import filters, morphology

try:
    from flika.process.filters import gaussian_blur as flika_gaussian_blur
    from flika.global_vars import g
    FLIKA_AVAILABLE = True
except ImportError:
    FLIKA_AVAILABLE = False

class ImageProcessor:
    @staticmethod
    def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian blur to the image.

        Args:
            image (np.ndarray): Input image.
            sigma (float): Standard deviation for Gaussian kernel.

        Returns:
            np.ndarray: Blurred image.

        Raises:
            ValueError: If sigma is not positive.
        """
        if sigma <= 0:
            raise ValueError("Sigma must be positive")

        if FLIKA_AVAILABLE:
            try:
                # Store the original g.m
                original_g_m = g.m
                # Set g.m to None to avoid the statusBar error
                g.m = None
                blurred = flika_gaussian_blur(image, sigma)
                # Restore the original g.m
                g.m = original_g_m
                return blurred
            except Exception as e:
                print(f"Error using Flika's gaussian_blur: {e}. Falling back to scipy.")
                return ndimage.gaussian_filter(image, sigma)
        else:
            return ndimage.gaussian_filter(image, sigma)

    @staticmethod
    def median_filter(image: np.ndarray, size: int) -> np.ndarray:
        """
        Apply median filter to the image.

        Args:
            image (np.ndarray): Input image.
            size (int): The size of the median filter window.

        Returns:
            np.ndarray: Filtered image.

        Raises:
            ValueError: If size is not positive odd integer.
        """
        if size <= 0 or size % 2 == 0:
            raise ValueError("Size must be a positive odd integer")

        return ndimage.median_filter(image, size)

    @staticmethod
    def sobel_edge(image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel edge detection to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Edge detection result.
        """
        return filters.sobel(image)

    @staticmethod
    def threshold(image: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's thresholding to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Thresholded binary image.
        """
        threshold = filters.threshold_otsu(image)
        return image > threshold

    @staticmethod
    def erode(image: np.ndarray) -> np.ndarray:
        """
        Apply erosion to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Eroded image.
        """
        return morphology.erosion(image)

    @staticmethod
    def dilate(image: np.ndarray) -> np.ndarray:
        """
        Apply dilation to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Dilated image.
        """
        return morphology.dilation(image)

    @staticmethod
    def z_project(image: np.ndarray, method: str = 'max') -> np.ndarray:
        """
        Perform Z-projection on the image stack.

        Args:
            image (np.ndarray): 3D image stack.
            method (str): Projection method, either 'max' or 'mean'.

        Returns:
            np.ndarray: Projected 2D image.

        Raises:
            ValueError: If the projection method is invalid.
        """
        if method not in ['max', 'mean']:
            raise ValueError("Method must be either 'max' or 'mean'")

        if method == 'max':
            return np.max(image, axis=0)
        else:
            return np.mean(image, axis=0)
