#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:52:38 2024

@author: george
"""

# analysis_steps.py

import numpy as np
import pandas as pd
import os
import glob
from sklearn.neighbors import KDTree
import logging
import tifffile  # or from skimage import io
from scipy import stats, spatial
import traceback
import warnings
import time
import signal

# class TimeoutError(Exception):
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutError("Function call timed out")

class AnalysisStep:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, data):
        raise NotImplementedError("Subclass must implement abstract method")

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)

    def log_error(self, message):
        self.logger.error(message)
        self.logger.error(traceback.format_exc())


class AddIDsToLocs(AnalysisStep):
    def run(self, data):
        self.log("Adding IDs to localization files")
        locs_list = glob.glob(os.path.join(self.config.folder_path, '**/*_locs.csv'), recursive=True)

        if not locs_list:
            self.log("No '_locs.csv' files found. Stopping analysis.", level=logging.WARNING)
            return None  # Return None to indicate that the analysis should stop

        for file in locs_list:
            try:
                df = pd.read_csv(file)
                df['id'] = df['id'].astype('int')
                df['frame'] = df['frame'].astype('int') - 1  # resetting first frame to zero to match flika display
                output_file = file.split('.csv')[0] + 'ID.csv'
                df.to_csv(output_file, index=False)
                self.log(f"Processed file: {file}")
            except Exception as e:
                self.log(f"Error processing file {file}: {str(e)}", level=logging.ERROR)
        return locs_list



class LinkPoints(AnalysisStep):
    def run(self, data):
        self.log("Linking points")
        tiff_list = glob.glob(os.path.join(self.config.folder_path, '**/*.tif'), recursive=True)
        processed_files = []
        for tiff_file in tiff_list:
            try:
                points_file = os.path.splitext(tiff_file)[0] + '_locsID.csv'
                if not os.path.exists(points_file):
                    self.log(f"Points file not found for {tiff_file}", level=logging.WARNING)
                    continue

                points = self.load_points(points_file)
                if points is None or len(points) == 0:
                    self.log(f"No valid points loaded from {points_file}", level=logging.WARNING)
                    continue

                linked_points = self.link_points(points)
                if linked_points is None or len(linked_points) == 0:
                    self.log(f"No linked points for {tiff_file}", level=logging.WARNING)
                    continue

                tracks_file = os.path.splitext(points_file)[0] + '_tracks.csv'
                self.save_tracks(linked_points, tracks_file)
                self.log(f"Linked points for file: {tiff_file}")
                processed_files.append(tracks_file)
            except Exception as e:
                self.log_error(f"Error linking points for file {tiff_file}: {str(e)}")
        return processed_files

    def load_points(self, filename):
        try:
            points_df = pd.read_csv(filename)
            if points_df.empty:
                self.log(f"Empty dataframe loaded from {filename}", level=logging.WARNING)
                return None

            points_df['frame'] = points_df['frame'].astype(int)
            if 'x [nm]' in points_df.columns and 'y [nm]' in points_df.columns:
                points_df['x'] = points_df['x [nm]'] / self.config.pixel_size
                points_df['y'] = points_df['y [nm]'] / self.config.pixel_size
            elif 'x' not in points_df.columns or 'y' not in points_df.columns:
                self.log(f"Required columns 'x' and 'y' not found in {filename}", level=logging.ERROR)
                return None
            points = points_df[['frame', 'x', 'y']].values
            return points
        except Exception as e:
            self.log(f"Error loading points from {filename}: {str(e)}", level=logging.ERROR)
            self.log(f"Error details: {traceback.format_exc()}", level=logging.DEBUG)
            return None

    def link_points(self, points):
        if len(points) == 0:
            self.log("No points to link", level=logging.WARNING)
            return None

        frames = np.unique(points[:, 0]).astype(int)
        max_frame = int(np.max(frames))
        pts_by_frame = [points[points[:, 0] == frame, 1:] for frame in range(max_frame + 1)]

        tracks = []
        for frame in frames:
            if frame >= len(pts_by_frame):
                self.log(f"Frame {frame} out of range", level=logging.WARNING)
                continue
            for pt_idx in range(len(pts_by_frame[frame])):
                if not any(len(track) > 0 and track[-1][0] == frame - 1 and track[-1][1] == pt_idx for track in tracks):
                    track = [(frame, pt_idx)]
                    track = self.extend_track(track, pts_by_frame, frame)
                    tracks.append(track)

        return self.create_tracks_array(points, tracks, pts_by_frame)

    def extend_track(self, track, pts_by_frame, current_frame):
        max_frames_skipped = self.config.max_frames_skipped
        max_distance = self.config.max_distance

        for dt in range(1, max_frames_skipped + 2):
            next_frame = current_frame + dt
            if next_frame >= len(pts_by_frame):
                self.log(f"Reached end of frames at frame {next_frame}", level=logging.DEBUG)
                return track

            if len(pts_by_frame[current_frame]) == 0:
                self.log(f"No points in current frame {current_frame}", level=logging.DEBUG)
                return track

            current_pt = pts_by_frame[current_frame][track[-1][1]]
            next_pts = pts_by_frame[next_frame]

            if len(next_pts) == 0:
                self.log(f"No points in next frame {next_frame}", level=logging.DEBUG)
                continue

            distances = np.sqrt(np.sum((next_pts - current_pt) ** 2, axis=1))
            if np.any(distances < max_distance):
                next_pt_idx = np.argmin(distances)
                track.append((next_frame, next_pt_idx))
                return self.extend_track(track, pts_by_frame, next_frame)

        self.log(f"Could not extend track beyond frame {current_frame}", level=logging.DEBUG)
        return track

    def create_tracks_array(self, points, tracks, pts_by_frame):
        tracked_points = []
        for track_id, track in enumerate(tracks):
            for frame, pt_idx in track:
                if frame >= len(pts_by_frame):
                    self.log(f"Frame {frame} out of range in track {track_id}", level=logging.WARNING)
                    continue
                if pt_idx >= len(pts_by_frame[frame]):
                    self.log(f"Point index {pt_idx} out of range in frame {frame}, track {track_id}", level=logging.WARNING)
                    continue
                point_index = np.where((points[:, 0] == frame) &
                                       (points[:, 1] == pts_by_frame[frame][pt_idx][0]) &
                                       (points[:, 2] == pts_by_frame[frame][pt_idx][1]))[0]
                if len(point_index) > 0:
                    tracked_points.append(np.append(points[point_index[0]], track_id))
                else:
                    self.log(f"Point not found for frame {frame}, pt_idx {pt_idx}, track {track_id}", level=logging.WARNING)
        return np.array(tracked_points)

    def save_tracks(self, tracks, filename):
        try:
            df = pd.DataFrame(tracks, columns=['frame', 'x', 'y', 'track_id'])
            df.to_csv(filename, index=False)
            self.log(f"Saved tracks to {filename}")
        except Exception as e:
            self.log.error(f"Error saving tracks to {filename}: {str(e)}")

class CalculateFeatures(AnalysisStep):
    def run(self, data):
        self.log("Calculating features")
        tracks_list = glob.glob(os.path.join(self.config.folder_path, '**/*_tracks.csv'), recursive=True)
        for track_file in tracks_list:
            try:
                df = pd.read_csv(track_file)
                df = self.calculate_features(df)
                output_file = os.path.splitext(track_file)[0] + '_features.csv'
                df.to_csv(output_file, index=False)
                self.log(f"Calculated features for file: {track_file}")
            except Exception as e:
                self.log_error(f"Error calculating features for file {track_file}: {str(e)}")
        return tracks_list

    def calculate_features(self, df):
        grouped = df.groupby('track_id')

        features = grouped.apply(self.calculate_track_features)
        df = df.merge(features, on='track_id')

        return df

    def calculate_track_features(self, track):
        points = track[['x', 'y']].values

        if len(points) < 3:
            return pd.Series({
                'radius_gyration': np.nan,
                'asymmetry': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'fractal_dimension': np.nan,
                'net_displacement': np.nan,
                'efficiency': np.nan,
                'sin_mean': np.nan,
                'cos_mean': np.nan
            })

        rg, asymmetry, skewness, kurtosis = self.radius_gyration_asymmetry_skewness_kurtosis(points)
        fractal_dim = self.fractal_dimension(points)
        net_displacement, efficiency = self.net_displacement_efficiency(points)
        sin_mean, cos_mean = self.summed_sines_cosines(points)

        return pd.Series({
            'radius_gyration': rg,
            'asymmetry': asymmetry,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'fractal_dimension': fractal_dim,
            'net_displacement': net_displacement,
            'efficiency': efficiency,
            'sin_mean': sin_mean,
            'cos_mean': cos_mean
        })


    def radius_gyration_asymmetry_skewness_kurtosis(self, points):
        center = points.mean(0)
        normed_points = points - center
        rg_tensor = np.einsum('im,in->mn', normed_points, normed_points) / len(points)
        eig_values, eig_vectors = np.linalg.eig(rg_tensor)

        rg = np.sqrt(np.sum(eig_values))

        try:
            asymmetry = -np.log(1 - (np.diff(eig_values)**2 / (2 * np.sum(eig_values)**2)))
        except (RuntimeWarning, ValueError):
            asymmetry = np.nan

        dom_eig_vect = eig_vectors[:, np.argmax(eig_values)]
        proj = np.dot(np.diff(points, axis=0), dom_eig_vect)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skewness = stats.skew(proj)
            kurtosis = stats.kurtosis(proj)

        return rg, asymmetry, skewness, kurtosis

    def fractal_dimension(self, points):
        # Check if points are collinear
        if len(points) < 3:
            return np.nan

        x0, y0 = points[0]
        slopes = [((y - y0) / (x - x0) if x != x0 else np.inf) for x, y in points[1:]]
        if all(s == slopes[0] for s in slopes):
            return np.nan

        total_path_length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        step_count = len(points)

        hull = spatial.ConvexHull(points)
        hull_points = points[hull.vertices]
        largest_distance = np.max(spatial.distance_matrix(hull_points, hull_points))

        return np.log(step_count) / np.log(step_count * largest_distance / total_path_length)

    def net_displacement_efficiency(self, points):
        net_displacement = np.linalg.norm(points[-1] - points[0])
        total_path_length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))

        try:
            efficiency = net_displacement**2 / ((len(points) - 1) * total_path_length**2)
        except ZeroDivisionError:
            efficiency = np.nan

        return net_displacement, efficiency

    def summed_sines_cosines(self, points):
        if len(points) < 3:
            return np.nan, np.nan

        vectors = np.diff(points, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_diffs = np.diff(angles)

        sin_vals = np.sin(angle_diffs)
        cos_vals = np.cos(angle_diffs)

        return np.mean(sin_vals), np.mean(cos_vals)

class AddNearestNeighbors(AnalysisStep):
    def run(self, data):
        self.log("Adding nearest neighbor information")
        file_list = glob.glob(os.path.join(self.config.folder_path, '**/*_features.csv'), recursive=True)
        for file in file_list:
            try:
                df = pd.read_csv(file)
                df = self.add_nn_info(df)
                output_file = os.path.splitext(file)[0] + '_nn.csv'
                df.to_csv(output_file, index=False)
                self.log(f"Added nearest neighbor info for file: {file}")
            except Exception as e:
                self.log_error(f"Error adding nearest neighbor info for file {file}: {str(e)}")
        return file_list

    def add_nn_info(self, df):
        df = df.sort_values(by=['frame'])
        nn_dist_list = []
        nn_index_list = []
        frames = df['frame'].unique()

        for frame in frames:
            frame_xy = df[df['frame'] == frame][['x', 'y']].values
            if len(frame_xy) > 1:
                distances, indexes = self.get_nearest_neighbors(frame_xy, frame_xy, k=2)
                nn_dist_list.extend(distances[:, 1])
                nn_index_list.extend(indexes[:, 1])
            else:
                nn_dist_list.extend([np.nan])
                nn_index_list.extend([-1])

        df['nn_distance'] = nn_dist_list
        df['nn_index'] = nn_index_list
        return df

    def get_nearest_neighbors(self, train, test, k=2):
        tree = KDTree(train, leaf_size=5)
        return tree.query(test, k=k)

class AddVelocity(AnalysisStep):
    def run(self, data):
        self.log("Adding velocity information")
        file_list = glob.glob(os.path.join(self.config.folder_path, '**/*_nn.csv'), recursive=True)
        for file in file_list:
            try:
                df = pd.read_csv(file)
                df = self.add_velocity_info(df)
                output_file = os.path.splitext(file)[0] + '_velocity.csv'
                df.to_csv(output_file, index=False)
                self.log(f"Added velocity info for file: {file}")
            except Exception as e:
                self.log_error(f"Error adding velocity info for file {file}: {str(e)}")
        return file_list

    def add_velocity_info(self, df):
        df = df.sort_values(['track_id', 'frame'])
        df['dx'] = df.groupby('track_id')['x'].diff()
        df['dy'] = df.groupby('track_id')['y'].diff()
        df['dt'] = df.groupby('track_id')['frame'].diff()

        df['velocity'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']
        df['direction'] = np.arctan2(df['dy'], df['dx']) * 180 / np.pi

        df['mean_velocity'] = df.groupby('track_id')['velocity'].transform('mean')

        return df


class AddMissingPoints(AnalysisStep):
    def run(self, data):
        self.log("Adding missing points")
        file_list = glob.glob(os.path.join(self.config.folder_path, '**/*_velocity.csv'), recursive=True)
        for file in file_list:
            try:
                start_time = time.time()
                self.log(f"Processing file: {file}")
                df = pd.read_csv(file)
                self.log(f"Loaded dataframe with shape: {df.shape}")

                df = self.add_missing_points(df)

                if df is not None:
                    output_file = os.path.splitext(file)[0] + '_complete.csv'
                    df.to_csv(output_file, index=False)
                    end_time = time.time()
                    self.log(f"Added missing points for file: {file}. Time taken: {end_time - start_time:.2f} seconds")
                else:
                    self.log_error(f"Adding missing points timed out for file: {file}")
            except Exception as e:
                self.log_error(f"Error adding missing points for file {file}: {str(e)}")
        return file_list

    def add_missing_points(self, df, timeout=300):  # 5 minutes timeout
        start_time = time.time()
        self.log("Starting to add missing points")
        all_frames = np.arange(df['frame'].min(), df['frame'].max() + 1)

        complete_df = []
        total_tracks = df['track_id'].nunique()

        for i, (track_id, track_df) in enumerate(df.groupby('track_id')):
            if i % 100 == 0:  # Log progress every 100 tracks
                self.log(f"Processing track {i+1}/{total_tracks}")
                if time.time() - start_time > timeout:
                    self.log_error(f"Processing timed out after {timeout} seconds")
                    return None

            track_frames = track_df['frame'].values
            missing_frames = np.setdiff1d(all_frames, track_frames)

            if len(missing_frames) > 0:
                # Create a full frame index
                full_frames = np.union1d(track_frames, missing_frames)

                # Interpolate x and y values
                x_interp = np.interp(full_frames, track_frames, track_df['x'].values)
                y_interp = np.interp(full_frames, track_frames, track_df['y'].values)

                # Create a new dataframe for the full track
                full_track_df = pd.DataFrame({
                    'frame': full_frames,
                    'track_id': track_id,
                    'x': x_interp,
                    'y': y_interp
                })

                # Merge with original data to preserve other columns
                merged_track_df = pd.merge(full_track_df, track_df, on=['frame', 'track_id', 'x', 'y'], how='left')
                complete_df.append(merged_track_df)
            else:
                complete_df.append(track_df)

        result = pd.concat(complete_df).sort_values(['track_id', 'frame']).reset_index(drop=True)
        end_time = time.time()
        self.log(f"Finished adding missing points. Time taken: {end_time - start_time:.2f} seconds")
        self.log(f"Result dataframe shape: {result.shape}")
        return result

class AddBackgroundSubtractedIntensity(AnalysisStep):
    def __init__(self, config, microview_instance):
        super().__init__(config)
        self.microview = microview_instance

    def run(self, data):
        self.log("Adding background-subtracted intensity")
        file_list = glob.glob(os.path.join(self.config.folder_path, '**/*_complete.csv'), recursive=True)
        for file in file_list:
            try:
                df = pd.read_csv(file)
                tiff_file = os.path.splitext(file)[0].split('_locs')[0] + '.tif'
                roi_file = os.path.join(os.path.dirname(file), f"ROI_{os.path.basename(tiff_file).split('.')[0]}.txt")

                window = self.microview.flika_open_file(tiff_file)
                rois = self.microview.flika_open_rois(roi_file)

                df = self.add_bg_subtracted_intensity(df, window, rois[0])

                output_file = os.path.splitext(file)[0] + '_intensity.csv'
                df.to_csv(output_file, index=False)
                self.log(f"Added background-subtracted intensity for file: {file}")
            except Exception as e:
                self.log.error(f"Error adding background-subtracted intensity for file {file}: {str(e)}")
        return file_list

    def add_bg_subtracted_intensity(self, df, window, roi):
        image_data = window.image
        roi_trace = roi.getTrace()
        camera_estimate = np.min(image_data, axis=(1, 2))

        df['roi_1'] = np.interp(df['frame'], np.arange(len(roi_trace)), roi_trace)
        df['camera_black_estimate'] = np.interp(df['frame'], np.arange(len(camera_estimate)), camera_estimate)

        df['intensity'] = self.get_intensities(image_data, df[['frame', 'x', 'y']].values)
        df['intensity_bg_subtracted'] = df['intensity'] - df['roi_1']
        df['intensity_bg_and_camera_subtracted'] = df['intensity'] - df['roi_1'] - df['camera_black_estimate']

        return df

    def get_intensities(self, image_data, points):
        intensities = []
        for frame, x, y in points:
            frame, x, y = int(frame), int(x), int(y)
            x_min, x_max = max(0, x-1), min(image_data.shape[2], x+2)
            y_min, y_max = max(0, y-1), min(image_data.shape[1], y+2)
            intensities.append(np.mean(image_data[frame, y_min:y_max, x_min:x_max]))
        return intensities


class FilterTracks(AnalysisStep):
    def run(self, data):
        self.log("Filtering tracks")
        file_list = glob.glob(os.path.join(self.config.folder_path, '**/*_intensity.csv'), recursive=True)
        for file in file_list:
            try:
                df = pd.read_csv(file)
                df = self.filter_tracks(df)
                output_file = os.path.splitext(file)[0] + '_filtered.csv'
                df.to_csv(output_file, index=False)
                self.log(f"Filtered tracks for file: {file}")
            except Exception as e:
                self.log.error(f"Error filtering tracks for file {file}: {str(e)}")
        return file_list

    def filter_tracks(self, df):
        # Filter tracks based on minimum length and maximum velocity
        track_lengths = df.groupby('track_id').size()
        valid_tracks = track_lengths[track_lengths >= self.config.min_track_length].index
        df = df[df['track_id'].isin(valid_tracks)]

        df = df[df['velocity'] <= self.config.max_velocity]
        return df

class ClassifyTracks(AnalysisStep):
    def run(self, data):
        self.log("Classifying tracks")
        file_list = glob.glob(os.path.join(self.config.folder_path, '**/*_filtered.csv'), recursive=True)
        for file in file_list:
            try:
                df = pd.read_csv(file)
                df = self.classify_tracks(df)
                output_file = os.path.splitext(file)[0] + '_classified.csv'
                df.to_csv(output_file, index=False)
                self.log(f"Classified tracks for file: {file}")
            except Exception as e:
                self.log.error(f"Error classifying tracks for file {file}: {str(e)}")
        return file_list

    def classify_tracks(self, df):
        # Simple classification based on mean velocity and net displacement
        df['track_class'] = 'unknown'
        df.loc[df['mean_velocity'] > self.config.directed_velocity_threshold, 'track_class'] = 'directed'
        df.loc[(df['mean_velocity'] <= self.config.directed_velocity_threshold) &
               (df['net_displacement'] > self.config.confined_displacement_threshold), 'track_class'] = 'diffusive'
        df.loc[df['net_displacement'] <= self.config.confined_displacement_threshold, 'track_class'] = 'confined'
        return df

class VisualizeTracksCumulative(AnalysisStep):
    def run(self, data):
        self.log("Visualizing tracks cumulatively")
        file_list = glob.glob(os.path.join(self.config.folder_path, '**/*_classified.csv'), recursive=True)
        for file in file_list:
            try:
                df = pd.read_csv(file)
                self.visualize_tracks(df, file)
                self.log(f"Visualized tracks for file: {file}")
            except Exception as e:
                self.log.error(f"Error visualizing tracks for file {file}: {str(e)}")
        return file_list

    def visualize_tracks(self, df, file):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        for track_id, track_df in df.groupby('track_id'):
            plt.plot(track_df['x'], track_df['y'], alpha=0.5)

        plt.title("Cumulative Track Visualization")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.axis('equal')

        output_file = os.path.splitext(file)[0] + '_cumulative_tracks.png'
        plt.savefig(output_file)
        plt.close()
