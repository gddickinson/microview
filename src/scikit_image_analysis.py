#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:28:50 2024

@author: george
"""

# scikit_image_analysis.py

import numpy as np
import logging
from skimage import filters, morphology, feature, segmentation, measure, exposure, restoration, transform, util
from scipy import ndimage
from typing import Union, Tuple, List, Dict
from skimage import filters, exposure, restoration, feature
from skimage.filters import hessian
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import gaussian_gradient_magnitude as ggm
import inspect

import skimage
print(f"scikit-image version: {skimage.__version__}")

logger = logging.getLogger(__name__)

class ScikitImageAnalysis:
    def __init__(self):
        self.image = None
        self.is_time_series = False

    def set_image(self, image: np.ndarray):
        self.image = image
        self.is_time_series = image.ndim > 2 and image.shape[0] > 1
        logger.info(f"Set image with shape {image.shape}, time series: {self.is_time_series}")
        print(f"Image stats - Min: {np.min(image)}, Max: {np.max(image)}, Mean: {np.mean(image)}")

    def apply_to_all_frames(self, func, *args, **kwargs):
        if self.is_time_series:
            result = []
            for i, frame in enumerate(self.image):
                try:
                    result.append(func(frame, *args, **kwargs))
                except Exception as e:
                    print(f"Error processing frame {i}: {str(e)}")
                    result.append(np.zeros_like(frame))
            result = np.array(result)
        else:
            try:
                result = func(self.image, *args, **kwargs)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                result = np.zeros_like(self.image)

        if result.size > 0:
            print(f"apply_to_all_frames result - Shape: {result.shape}, Min: {np.min(result)}, Max: {np.max(result)}, Mean: {np.mean(result)}")
        else:
            print(f"apply_to_all_frames result - Empty array with shape: {result.shape}")

        return result

    # Filtering methods
    def gaussian_filter(self, sigma=1.0):
        result = self.apply_to_all_frames(filters.gaussian, sigma=sigma)
        print(f"Gaussian filter result - Shape: {result.shape}, Min: {np.min(result)}, Max: {np.max(result)}, Mean: {np.mean(result)}")
        return result.astype(np.float32)  # Ensure the result is float32


    def median_filter(self, size: int = 3) -> np.ndarray:
        return self.apply_to_all_frames(filters.median, footprint=np.ones((size, size)))

    def bilateral_filter(self, sigma_color: float = 0.1, sigma_spatial: float = 1) -> np.ndarray:
        return self.apply_to_all_frames(restoration.denoise_bilateral, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

    # Edge detection methods
    def sobel_edge(self) -> np.ndarray:
        return self.apply_to_all_frames(filters.sobel)

    def canny_edge(self, sigma: float = 1.0) -> np.ndarray:
        return self.apply_to_all_frames(feature.canny, sigma=sigma)

    # Thresholding methods
    def otsu_threshold(self) -> np.ndarray:
        return self.apply_to_all_frames(lambda img: img > filters.threshold_otsu(img))

    def adaptive_threshold(self, block_size: int = 35, offset: float = 0) -> np.ndarray:
        return self.apply_to_all_frames(lambda img: img > filters.threshold_local(img, block_size, offset=offset))

    # Morphological operations
    def erode(self, size: int = 3) -> np.ndarray:
        footprint = morphology.disk(size) if self.image.ndim == 2 or (self.is_time_series and len(self.image.shape[1:]) == 2) else morphology.ball(size)
        return self.apply_to_all_frames(morphology.erosion, footprint=footprint)

    def dilate(self, size: int = 3) -> np.ndarray:
        footprint = morphology.disk(size) if self.image.ndim == 2 or (self.is_time_series and len(self.image.shape[1:]) == 2) else morphology.ball(size)
        return self.apply_to_all_frames(morphology.dilation, footprint=footprint)

    def open(self, size: int = 3) -> np.ndarray:
        footprint = morphology.disk(size) if self.image.ndim == 2 or (self.is_time_series and len(self.image.shape[1:]) == 2) else morphology.ball(size)
        return self.apply_to_all_frames(morphology.opening, footprint=footprint)

    def close(self, size: int = 3) -> np.ndarray:
        footprint = morphology.disk(size) if self.image.ndim == 2 or (self.is_time_series and len(self.image.shape[1:]) == 2) else morphology.ball(size)
        return self.apply_to_all_frames(morphology.closing, footprint=footprint)

    # Segmentation methods
    def watershed_segmentation(self, markers: Union[np.ndarray, int]) -> np.ndarray:
        def segment(img):
            if isinstance(markers, int):
                # Use peak_local_max to find local maxima
                coordinates = feature.peak_local_max(img, num_peaks=markers, footprint=np.ones((3, 3)))

                # Create a boolean mask
                local_max_mask = np.zeros(img.shape, dtype=bool)
                local_max_mask[tuple(coordinates.T)] = True

                # Label the mask
                local_markers = measure.label(local_max_mask)
            else:
                local_markers = markers

            return segmentation.watershed(-img, local_markers, mask=img)

        return self.apply_to_all_frames(segment)

    # Feature detection
    def detect_blobs(self, min_sigma: float = 1, max_sigma: float = 30, num_sigma: int = 10, threshold: float = 0.1) -> np.ndarray:
        def detect(img):
            blobs = feature.blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
            return blobs  # This is already a numpy array
        return self.apply_to_all_frames(detect)

    # Measurements
    def measure_properties(self, labeled_image: np.ndarray) -> List[Dict]:
        props = measure.regionprops(labeled_image)
        return [{
            'label': p.label,
            'area': p.area,
            'centroid': p.centroid,
            'bbox': p.bbox,
            'eccentricity': p.eccentricity,
            'mean_intensity': p.mean_intensity,
            'orientation': p.orientation
        } for p in props]

    # Image enhancement
    def adjust_gamma(self, gamma: float = 1.0) -> np.ndarray:
        return self.apply_to_all_frames(exposure.adjust_gamma, gamma=gamma)

    def equalize_histogram(self) -> np.ndarray:
        return self.apply_to_all_frames(exposure.equalize_hist)

    # Transformations
    def resize(self, output_shape: Tuple[int, int]) -> np.ndarray:
        return self.apply_to_all_frames(transform.resize, output_shape=output_shape)

    def rotate(self, angle: float) -> np.ndarray:
        return self.apply_to_all_frames(transform.rotate, angle=angle)

    # Time series specific methods
    def temporal_mean(self) -> np.ndarray:
        if not self.is_time_series:
            logger.warning("Temporal mean called on non-time series data")
            return self.image
        return np.mean(self.image, axis=0)

    def temporal_max(self) -> np.ndarray:
        if not self.is_time_series:
            logger.warning("Temporal max called on non-time series data")
            return self.image
        return np.max(self.image, axis=0)

    def temporal_std(self) -> np.ndarray:
        if not self.is_time_series:
            logger.warning("Temporal std called on non-time series data")
            return np.zeros_like(self.image)
        return np.std(self.image, axis=0)

    # Particle tracking
    def track_particles(self, max_distance: float = 10, min_trajectory_length: int = 5) -> List[Dict]:
        if not self.is_time_series:
            logger.warning("Particle tracking called on non-time series data")
            return []

        from trackpy import locate, link

        def locate_particles(frame):
            return locate(frame, diameter=11, minmass=100)

        particles = [locate_particles(frame) for frame in self.image]
        trajectories = link(particles, search_range=max_distance, memory=3)

        filtered_trajectories = trajectories[trajectories.groupby('particle').size() >= min_trajectory_length]

        return [{'particle': p, 'trajectory': group[['frame', 'x', 'y']].values.tolist()}
                for p, group in filtered_trajectories.groupby('particle')]

    def unsharp_mask(self, radius=1, amount=1):
        return self.apply_to_all_frames(filters.unsharp_mask, radius=radius, amount=amount)

    def gaussian_gradient_magnitude(self, sigma=1):
        return self.apply_to_all_frames(ggm, sigma=sigma)

    def laplacian(self):
        return self.apply_to_all_frames(filters.laplace)

    def frangi(self, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15, black_ridges=True):
        # Get the parameters of the frangi function
        frangi_params = inspect.signature(filters.frangi).parameters

        # Create a dictionary of parameters
        kwargs = {'scale_range': scale_range, 'scale_step': scale_step, 'black_ridges': black_ridges}

        # Add beta1 and beta2 only if they are accepted by the function
        if 'beta1' in frangi_params:
            kwargs['beta1'] = beta1
        if 'beta2' in frangi_params:
            kwargs['beta2'] = beta2

        return self.apply_to_all_frames(filters.frangi, **kwargs)

    def tophat(self, size=5):
        print(f"Image shape in tophat: {self.image.shape}")
        def apply_tophat(image):
            if image.ndim == 2:
                selem = morphology.disk(size)
            elif image.ndim == 3:
                selem = morphology.ball(size)
            else:
                raise ValueError(f"Unsupported image dimension: {image.ndim}")
            return morphology.white_tophat(image, footprint=selem)
        return self.apply_to_all_frames(apply_tophat)

    def bottomhat(self, size=5):
        print(f"Image shape in bottomhat: {self.image.shape}")
        def apply_bottomhat(image):
            if image.ndim == 2:
                selem = morphology.disk(size)
            elif image.ndim == 3:
                selem = morphology.ball(size)
            else:
                raise ValueError(f"Unsupported image dimension: {image.ndim}")
            return morphology.black_tophat(image, footprint=selem)
        return self.apply_to_all_frames(apply_bottomhat)

    def local_binary_pattern(self, P=8, R=1, method='uniform'):
        return self.apply_to_all_frames(feature.local_binary_pattern, P=P, R=R, method=method)

    def contrast_stretch(self, in_range=(0, 100), out_range=(0, 255)):
        return self.apply_to_all_frames(exposure.rescale_intensity, in_range=in_range, out_range=out_range)

    def denoise_nl_means(self, patch_size=5, patch_distance=6, h=0.1):
        return self.apply_to_all_frames(restoration.denoise_nl_means, patch_size=patch_size, patch_distance=patch_distance, h=h)

    def hessian_matrix_eigvals(self, sigma=1):
        def apply_hessian(image):
            try:
                hessian_matrices = hessian_matrix(image, sigma=sigma, use_gaussian_derivatives=True)
                eigvals = hessian_matrix_eigvals(hessian_matrices)

                # We'll return the larger absolute eigenvalue
                return np.max(np.abs(eigvals), axis=0)
            except Exception as e:
                print(f"Error in hessian_matrix_eigvals: {str(e)}")
                return np.zeros_like(image)
        return self.apply_to_all_frames(apply_hessian)

    # Add more methods as needed...
