#!/usr/bin/env python3
"""
Generalized Zonal Statistics Tool with Monte Carlo Uncertainty Analysis
Supports any mean/std input pairs for uncertainty estimation
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.session import AWSSession
from rasterio.crs import CRS
from rio_tiler.io import COGReader
from rio_tiler.mosaic import mosaic_reader
from rio_tiler.mosaic.methods import defaults
from shapely.geometry import mapping, shape
from scipy import stats as scipy_stats
import boto3

# Import your existing utilities
try:
    from CovariateUtils import get_rio_aws_session_from_creds, local_to_s3, reader
except ImportError:
    print("Warning: CovariateUtils not available. Some S3 functionality may be limited.")

# Import trend functions
try:
    from age_trend_functions import (
        get_kendall_trend_class_labels, 
        get_trend_category_mapping, 
        remap_trend_classes,
        classify_age_cohorts,
        get_default_age_classification
    )
    HAS_TREND_FUNCTIONS = True
except ImportError:
    print("Warning: age_trend_functions not available. Trend class functionality disabled.")
    HAS_TREND_FUNCTIONS = False

warnings.filterwarnings('ignore')

def monte_carlo_estimation(mean_data, std_data, pixel_area_ha=0.09, 
                          n_simulations=1000, confidence_level=0.95):
    """
    General Monte Carlo estimation for any mean/std data pairs
    
    Parameters:
    -----------
    mean_data : array
        Mean values for each pixel
    std_data : array  
        Standard deviation values for each pixel
    pixel_area_ha : float
        Area of each pixel in hectares (for scaling)
    n_simulations : int
        Number of Monte Carlo simulations
    confidence_level : float
        Confidence level for uncertainty bounds (e.g., 0.95 for 95% CI)
    """
    
    n_pixels = len(mean_data)
    total_sims = np.zeros(n_simulations)
    
    # For each simulation
    for sim in range(n_simulations):
        # Sample from normal distribution for each pixel
        # Truncate at zero (no negative values)
        pixel_samples = np.maximum(0, 
            np.random.normal(mean_data, std_data))
        
        # Scale by pixel area and sum
        total_value = np.sum(pixel_samples * pixel_area_ha)
        total_sims[sim] = total_value
    
    # Calculate statistics
    mean_estimate = np.mean(total_sims)
    std_estimate = np.std(total_sims)
    
    # Confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(total_sims, lower_percentile)
    ci_upper = np.percentile(total_sims, upper_percentile)
    
    # Coefficient of variation
    cv = std_estimate / mean_estimate if mean_estimate != 0 else 0
    
    return {
        'mc_mean': mean_estimate,
        'mc_std': std_estimate,
        'mc_lower_ci': ci_lower,
        'mc_upper_ci': ci_upper,
        'mc_cv': cv,
        'mc_count': n_pixels
    }

def analytical_estimation(mean_data, std_data, pixel_area_ha=0.09):
    """
    Analytical estimation using error propagation
    """
    
    # Scale by pixel area
    mean_scaled = mean_data * pixel_area_ha
    std_scaled = std_data * pixel_area_ha
    
    # Sum across pixels
    total_mean = np.sum(mean_scaled)
    
    # Uncertainty propagation for independent pixels
    # Variance of sum = sum of variances
    total_variance = np.sum(std_scaled**2)
    total_std = np.sqrt(total_variance)
    
    # 95% confidence interval (assuming normal)
    ci_95 = 1.96 * total_std
    cv = total_std / total_mean if total_mean != 0 else 0
    
    return {
        'analytical_mean': total_mean,
        'analytical_std': total_std,
        'analytical_lower_ci': total_mean - ci_95,
        'analytical_upper_ci': total_mean + ci_95,
        'analytical_cv': cv,
        'analytical_count': len(mean_data)
    }

def calculate_uncertainty_stats(mean_data, std_data, age_data=None, trend_data=None, 
                               age_classes=None, pixel_area_ha=0.09, method='monte_carlo'):
    """
    Calculate uncertainty statistics with age and trend class breakdowns
    """
    
    stats = {}
    
    # Overall uncertainty estimation
    if method == 'monte_carlo':
        overall_stats = monte_carlo_estimation(mean_data, std_data, pixel_area_ha)
    else:
        overall_stats = analytical_estimation(mean_data, std_data, pixel_area_ha)
    
    # Add to results with prefixes
    for key, value in overall_stats.items():
        stats[f"total_{key}"] = value
    
    # Age class analysis
    if age_classes and age_data is not None:
        valid_mask = ~np.isnan(mean_data) & ~np.isnan(std_data) & ~np.isnan(age_data)
        valid_mean = mean_data[valid_mask]
        valid_std = std_data[valid_mask]
        valid_age = age_data[valid_mask]
        
        for age_class, (min_age, max_age) in age_classes.items():
            if min_age is None:
                age_mask = valid_age <= max_age
            elif max_age is None:
                age_mask = valid_age >= min_age
            else:
                age_mask = (valid_age >= min_age) & (valid_age <= max_age)
            
            if np.sum(age_mask) > 0:
                age_mean = valid_mean[age_mask]
                age_std = valid_std[age_mask]
                
                if method == 'monte_carlo':
                    age_stats = monte_carlo_estimation(age_mean, age_std, pixel_area_ha)
                else:
                    age_stats = analytical_estimation(age_mean, age_std, pixel_area_ha)
                
                # Add age-specific stats
                for key, value in age_stats.items():
                    stats[f"{age_class}_{key}"] = value
            else:
                # No pixels in this age class
                if method == 'monte_carlo':
                    null_stats = ['mc_mean', 'mc_std', 'mc_lower_ci', 'mc_upper_ci', 'mc_cv']
                    for stat in null_stats:
                        stats[f"{age_class}_{stat}"] = np.nan
                    stats[f"{age_class}_mc_count"] = 0
                else:
                    null_stats = ['analytical_mean', 'analytical_std', 'analytical_lower_ci', 'analytical_upper_ci', 'analytical_cv']
                    for stat in null_stats:
                        stats[f"{age_class}_{stat}"] = np.nan
                    stats[f"{age_class}_analytical_count"] = 0
    
    # Trend class analysis
    if trend_data is not None and HAS_TREND_FUNCTIONS:
        valid_mask = ~np.isnan(mean_data) & ~np.isnan(std_data) & ~np.isnan(trend_data)
        valid_mean = mean_data[valid_mask]
        valid_std = std_data[valid_mask]
        
        # Remap trend classes
        simplified_trends = remap_trend_classes(trend_data[valid_mask])
        trend_category_names = {
            0: 'strong_decline', 1: 'moderate_decline', 2: 'stable',
            3: 'moderate_increase', 4: 'strong_increase'
        }
        
        for category_val, category_name in trend_category_names.items():
            trend_mask = (simplified_trends == category_val)
            
            if np.sum(trend_mask) > 0:
                trend_mean = valid_mean[trend_mask]
                trend_std = valid_std[trend_mask]
                
                if method == 'monte_carlo':
                    trend_stats = monte_carlo_estimation(trend_mean, trend_std, pixel_area_ha)
                else:
                    trend_stats = analytical_estimation(trend_mean, trend_std, pixel_area_ha)
                
                # Add trend-specific stats
                for key, value in trend_stats.items():
                    stats[f"trend_{category_name}_{key}"] = value
            else:
                # No pixels in this trend class
                if method == 'monte_carlo':
                    null_stats = ['mc_mean', 'mc_std', 'mc_lower_ci', 'mc_upper_ci', 'mc_cv']
                    for stat in null_stats:
                        stats[f"trend_{category_name}_{stat}"] = np.nan
                    stats[f"trend_{category_name}_mc_count"] = 0
                else:
                    null_stats = ['analytical_mean', 'analytical_std', 'analytical_lower_ci', 'analytical_upper_ci', 'analytical_cv']
                    for stat in null_stats:
                        stats[f"trend_{category_name}_{stat}"] = np.nan
                    stats[f"trend_{category_name}_analytical_count"] = 0
    
    # Age × Trend combinations
    if age_classes and age_data is not None and trend_data is not None and HAS_TREND_FUNCTIONS:
        valid_mask = (~np.isnan(mean_data) & ~np.isnan(std_data) & 
                     ~np.isnan(age_data) & ~np.isnan(trend_data))
        
        if np.sum(valid_mask) > 0:
            valid_mean_data = mean_data[valid_mask]
            valid_std_data = std_data[valid_mask]
            valid_age_data = age_data[valid_mask]
            valid_trend_data = trend_data[valid_mask]
            simplified_trends = remap_trend_classes(valid_trend_data)
            
            trend_category_names = {
                0: 'strong_decline', 1: 'moderate_decline', 2: 'stable',
                3: 'moderate_increase', 4: 'strong_increase'
            }
            
            for age_class_name, (min_age, max_age) in age_classes.items():
                if min_age is None:
                    age_mask = valid_age_data <= max_age
                elif max_age is None:
                    age_mask = valid_age_data >= min_age
                else:
                    age_mask = (valid_age_data >= min_age) & (valid_age_data <= max_age)
                
                for category_val, category_name in trend_category_names.items():
                    trend_mask = (simplified_trends == category_val)
                    combined_mask = age_mask & trend_mask
                    
                    if np.sum(combined_mask) > 0:
                        combo_mean = valid_mean_data[combined_mask]
                        combo_std = valid_std_data[combined_mask]
                        
                        if method == 'monte_carlo':
                            combo_stats = monte_carlo_estimation(combo_mean, combo_std, pixel_area_ha)
                        else:
                            combo_stats = analytical_estimation(combo_mean, combo_std, pixel_area_ha)
                        
                        # Add age×trend-specific stats
                        for key, value in combo_stats.items():
                            stats[f"{age_class_name}_trend_{category_name}_{key}"] = value
                    else:
                        # No pixels in this age×trend combination
                        if method == 'monte_carlo':
                            null_stats = ['mc_mean', 'mc_std', 'mc_lower_ci', 'mc_upper_ci', 'mc_cv']
                            for stat in null_stats:
                                stats[f"{age_class_name}_trend_{category_name}_{stat}"] = np.nan
                            stats[f"{age_class_name}_trend_{category_name}_mc_count"] = 0
                        else:
                            null_stats = ['analytical_mean', 'analytical_std', 'analytical_lower_ci', 'analytical_upper_ci', 'analytical_cv']
                            for stat in null_stats:
                                stats[f"{age_class_name}_trend_{category_name}_{stat}"] = np.nan
                            stats[f"{age_class_name}_trend_{category_name}_analytical_count"] = 0
    
    return stats

def calculate_basic_stats(data, name_prefix=""):
    """
    Calculate basic statistics (mean, std, count only - no sum)
    """
    if data is None or len(data) == 0:
        return {}
    
    # Remove nodata values
    valid_mask = (data != -9999) & (~np.isnan(data))
    valid_data = data[valid_mask]
    
    prefix = f"{name_prefix}_" if name_prefix else ""
    
    if len(valid_data) == 0:
        return {
            f'{prefix}count': 0,
            f'{prefix}mean': np.nan,
            f'{prefix}std': np.nan
        }
    
    return {
        f'{prefix}count': len(valid_data),
        f'{prefix}mean': np.mean(valid_data),
        f'{prefix}std': np.std(valid_data)
    }

def get_trend_class_stats(trend_data, age_data=None, age_classes=None):
    """
    Calculate trend class statistics, optionally by age class
    """
    if not HAS_TREND_FUNCTIONS:
        return {}
    
    # Get trend labels and mapping
    kendall_labels = get_kendall_trend_class_labels()
    
    # Remap to simplified trend categories
    simplified_trends = remap_trend_classes(trend_data)
    
    # Define simplified trend category names
    trend_category_names = {
        -1: 'no_data',
        0: 'strong_decline', 
        1: 'moderate_decline',
        2: 'stable',
        3: 'moderate_increase', 
        4: 'strong_increase'
    }
    
    trend_stats = {}
    
    # Overall trend class statistics
    valid_trend_mask = ~np.isnan(trend_data)
    valid_trend_data = trend_data[valid_trend_mask]
    total_pixels = len(valid_trend_data)
    
    if total_pixels > 0:
        # Count pixels in each original Kendall class
        for class_val, class_label in kendall_labels.items():
            count = np.sum(valid_trend_data == class_val)
            proportion = count / total_pixels if total_pixels > 0 else 0
            
            # Clean label for column name
            clean_label = class_label.lower().replace(' ', '_').replace('.', '')
            trend_stats[f"kendall_{clean_label}_count"] = count
            trend_stats[f"kendall_{clean_label}_prop"] = proportion
        
        # Count pixels in each simplified trend category
        valid_simplified = simplified_trends[valid_trend_mask]
        for category_val, category_name in trend_category_names.items():
            if category_val == -1:  # Skip no_data category
                continue
            count = np.sum(valid_simplified == category_val)
            proportion = count / total_pixels if total_pixels > 0 else 0
            
            trend_stats[f"trend_{category_name}_count"] = count
            trend_stats[f"trend_{category_name}_prop"] = proportion
    
    # Age-stratified trend statistics if age data provided
    if age_data is not None and age_classes is not None:
        if len(age_data) != len(trend_data):
            print(f"Warning: Age data size ({len(age_data)}) != trend data size ({len(trend_data)})")
            return trend_stats
        
        valid_both_mask = valid_trend_mask & ~np.isnan(age_data)
        valid_age_data = age_data[valid_both_mask]
        valid_simplified_for_age = simplified_trends[valid_both_mask]
        
        for age_class, (min_age, max_age) in age_classes.items():
            if min_age is None:
                age_mask = valid_age_data <= max_age
            elif max_age is None:
                age_mask = valid_age_data >= min_age
            else:
                age_mask = (valid_age_data >= min_age) & (valid_age_data <= max_age)
            
            age_simplified_trends = valid_simplified_for_age[age_mask]
            age_total_pixels = len(age_simplified_trends)
            
            if age_total_pixels > 0:
                for category_val, category_name in trend_category_names.items():
                    if category_val == -1:
                        continue
                    count = np.sum(age_simplified_trends == category_val)
                    proportion = count / age_total_pixels if age_total_pixels > 0 else 0
                    
                    trend_stats[f"{age_class}_trend_{category_name}_count"] = count
                    trend_stats[f"{age_class}_trend_{category_name}_prop"] = proportion
            else:
                for category_val, category_name in trend_category_names.items():
                    if category_val == -1:
                        continue
                    trend_stats[f"{age_class}_trend_{category_name}_count"] = 0
                    trend_stats[f"{age_class}_trend_{category_name}_prop"] = 0.0
    
    return trend_stats

class GeneralizedZonalStats:
    """Generalized zonal statistics processor using rio_tiler"""
    
    def __init__(self, aws_credentials=None, temp_dir=None):
        self.aws_session = None
        self.temp_dir = temp_dir or "/tmp"
        
        if aws_credentials:
            try:
                self.aws_session = get_rio_aws_session_from_creds(aws_credentials)
            except Exception as e:
                print(f"Warning: Could not initialize AWS session: {e}")
        
        # Set up AWS environment for public data
        os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
    
    def parse_band_spec(self, band_spec):
        """Parse band specification like '1', '1,3', or '1-3'"""
        if not band_spec or band_spec == 'all':
            return None
        
        bands = []
        for part in band_spec.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                bands.extend(range(start, end + 1))
            else:
                bands.append(int(part))
        
        return bands
    
    def load_tile_index_from_geojson(self, geojson_file: str, polygon_geometry, polygon_crs) -> Tuple[List[str], CRS]:
        """Load intersecting raster paths from GeoJSON tile index using spatial intersection"""
        
        try:
            # Read the tile index with geopandas
            if geojson_file.startswith('s3://'):
                from urllib.parse import urlparse
                parsed = urlparse(geojson_file)
                vsi_path = f"/vsis3/{parsed.netloc}{parsed.path}"
                tile_index = gpd.read_file(vsi_path)
            else:
                tile_index = gpd.read_file(geojson_file)
            
            # Reproject polygon to match tile index CRS if needed
            if polygon_crs != tile_index.crs:
                geom_gdf = gpd.GeoDataFrame([1], geometry=[polygon_geometry], crs=polygon_crs)
                geom_gdf = geom_gdf.to_crs(tile_index.crs)
                polygon_geometry = geom_gdf.geometry.iloc[0]
            
            # Find intersecting tiles
            intersecting_tiles = tile_index[tile_index.intersects(polygon_geometry)]
            
            # Extract s3_path from intersecting tiles
            raster_paths = []
            if 's3_path' in intersecting_tiles.columns:
                raster_paths = intersecting_tiles['s3_path'].dropna().tolist()
            elif 'local_path' in intersecting_tiles.columns:
                raster_paths = intersecting_tiles['local_path'].dropna().tolist()
            else:
                print(f"Warning: No 's3_path' or 'local_path' column found")
            
            # Get CRS from first raster file
            mosaic_crs = tile_index.crs
            if raster_paths:
                try:
                    with rasterio.env.Env(self.aws_session if self.aws_session else None):
                        with COGReader(raster_paths[0]) as cog:
                            mosaic_crs = cog.crs
                except Exception as e:
                    print(f"Warning: Could not read raster CRS: {e}")
            
            return raster_paths, mosaic_crs
            
        except Exception as e:
            print(f"Error loading tile index: {e}")
            return [], CRS.from_epsg(4326)
    
    def extract_band_data(self, geometry, raster_paths, mosaic_crs, band):
        """Extract data for a single band from rasters"""
        try:
            geom_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs='EPSG:4326')
            if geom_gdf.crs != mosaic_crs:
                geom_gdf = geom_gdf.to_crs(mosaic_crs)
            
            target_geometry = geom_gdf.geometry.iloc[0]
            bounds = target_geometry.bounds
            
            # Use adaptive resolution
            geom_area = target_geometry.area
            if geom_area > 1000000:
                res = 60
            elif geom_area > 100000:
                res = 30
            else:
                res = 15
                
            width = max(10, int((bounds[2] - bounds[0]) / res))
            height = max(10, int((bounds[3] - bounds[1]) / res))
            
            # Prevent excessive memory usage
            MAX_SIZE = 1500
            if width > MAX_SIZE or height > MAX_SIZE:
                scale_factor = min(MAX_SIZE/width, MAX_SIZE/height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
            
            # Create transform for this grid
            transform = rasterio.transform.from_bounds(*bounds, width, height)
            
            # Create geometry mask
            from rasterio.features import geometry_mask
            geom_mask = geometry_mask([mapping(target_geometry)], 
                                    out_shape=(height, width),
                                    transform=transform,
                                    invert=True)
            
            with rasterio.env.Env(self.aws_session if self.aws_session else None):
                def band_reader(src_path: str, *args, **kwargs):
                    with COGReader(src_path) as cog:
                        return cog.part(
                            bounds, 
                            bounds_crs=mosaic_crs,
                            dst_crs=mosaic_crs, 
                            height=height, 
                            width=width, 
                            indexes=(band,)
                        )
                
                img_data = mosaic_reader(
                    raster_paths,
                    band_reader,
                    pixel_selection=defaults.FirstMethod()
                )
                
                # Extract the data array
                if hasattr(img_data, 'array'):
                    data_array = img_data.array
                    data = data_array[0] if data_array.ndim == 3 else data_array
                elif hasattr(img_data, 'data'):
                    data_array = img_data.data
                    data = data_array[0] if data_array.ndim == 3 else data_array
                elif isinstance(img_data, tuple):
                    data_array = img_data[0].array if hasattr(img_data[0], 'array') else img_data[0].data
                    data = data_array[0] if data_array.ndim == 3 else data_array
                else:
                    data = img_data
                
                # Apply geometry mask
                masked_data = np.ma.masked_array(data, mask=~geom_mask)
                masked_data = np.ma.masked_invalid(masked_data)
                
                # Try to get nodata value and mask it
                try:
                    if hasattr(img_data, 'nodata') and img_data.nodata is not None:
                        masked_data = np.ma.masked_equal(masked_data, img_data.nodata)
                    elif isinstance(img_data, tuple) and hasattr(img_data[0], 'nodata') and img_data[0].nodata is not None:
                        masked_data = np.ma.masked_equal(masked_data, img_data[0].nodata)
                except:
                    pass
                    
                return masked_data
            
        except Exception as e:
            print(f"Error extracting band data: {e}")
            return None
    
    def extract_aligned_multi_data(self, geometry, datasets_config, polygon_crs):
        """
        Extract multiple datasets on the same spatial grid
        
        datasets_config: dict with structure:
        {
            'age': {'paths': [...], 'crs': CRS, 'band': int},
            'trend': {'paths': [...], 'crs': CRS, 'band': int}, 
            'mean_var1': {'paths': [...], 'crs': CRS, 'band': int},
            'std_var1': {'paths': [...], 'crs': CRS, 'band': int},
            ...
        }
        """
        
        try:
            # Use age data as reference CRS if available, otherwise first dataset
            ref_crs = None
            if 'age' in datasets_config:
                ref_crs = datasets_config['age']['crs']
            else:
                ref_crs = list(datasets_config.values())[0]['crs']
            
            # Convert geometry to reference CRS
            geom_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs=polygon_crs)
            if geom_gdf.crs != ref_crs:
                geom_gdf = geom_gdf.to_crs(ref_crs)
            
            target_geometry = geom_gdf.geometry.iloc[0]
            bounds = target_geometry.bounds
            
            # Calculate dimensions
            res = 30
            width = max(10, int((bounds[2] - bounds[0]) / res))
            height = max(10, int((bounds[3] - bounds[1]) / res))
            
            # Limit size
            MAX_SIZE = 1500
            if width > MAX_SIZE or height > MAX_SIZE:
                scale_factor = min(MAX_SIZE/width, MAX_SIZE/height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
            
            # Create transform and geometry mask
            transform = rasterio.transform.from_bounds(*bounds, width, height)
            from rasterio.features import geometry_mask
            geom_mask = geometry_mask([mapping(target_geometry)], 
                                     out_shape=(height, width),
                                     transform=transform,
                                     invert=True)
            
            # Extract data for each dataset
            extracted_data = {}
            
            for dataset_name, config in datasets_config.items():
                if not config['paths']:
                    continue
                
                # Calculate bounds for this dataset's CRS
                dataset_bounds = bounds
                if config['crs'] != ref_crs:
                    from rasterio.warp import transform_bounds
                    dataset_bounds = transform_bounds(ref_crs, config['crs'], *bounds)
                
                dataset_data = self._extract_single_dataset(
                    config['paths'], config['crs'], config['band'],
                    dataset_bounds, height, width, geom_mask,
                    dataset_name, target_crs=ref_crs
                )
                
                if dataset_data is not None:
                    extracted_data[dataset_name] = dataset_data
            
            # Apply common mask to all datasets
            if len(extracted_data) > 1:
                all_datasets = list(extracted_data.values())
                combined_mask = all_datasets[0].mask.copy()
                for dataset in all_datasets[1:]:
                    combined_mask = combined_mask | dataset.mask
                
                # Apply combined mask
                for name in extracted_data:
                    extracted_data[name] = np.ma.masked_array(
                        extracted_data[name].data, mask=combined_mask
                    )
            
            # Get compressed arrays
            extracted_valid = {}
            for name, data in extracted_data.items():
                extracted_valid[name] = data.compressed()
            
            return extracted_valid
            
        except Exception as e:
            print(f"Error in extract_aligned_multi_data: {e}")
            return {}
    
    def _extract_single_dataset(self, paths, crs, band, bounds, height, width, geom_mask, 
                               dataset_name, target_crs=None):
        """Helper function to extract a single dataset"""
        
        with rasterio.env.Env(self.aws_session if self.aws_session else None):
            try:
                def dataset_reader(src_path: str, *args, **kwargs):
                    with COGReader(src_path) as cog:
                        return cog.part(bounds, bounds_crs=crs, 
                                      dst_crs=target_crs or crs,
                                      height=height, width=width, indexes=(band,))
                
                img = mosaic_reader(paths, dataset_reader, pixel_selection=defaults.FirstMethod())
                
                if hasattr(img, 'array'):
                    array = img.array[0] if img.array.ndim == 3 else img.array
                elif hasattr(img, 'data'):
                    array = img.data[0] if img.data.ndim == 3 else img.data
                else:
                    array = img[0].array[0] if hasattr(img[0], 'array') else img[0].data[0]
                
                # Apply geometry mask and nodata mask
                masked = np.ma.masked_array(array, mask=~geom_mask)
                masked = np.ma.masked_invalid(masked)
                
                # Try to mask nodata values
                if hasattr(img, 'nodata') and img.nodata is not None:
                    masked = np.ma.masked_equal(masked, img.nodata)
                
                return masked
                
            except Exception as e:
                print(f"Error extracting {dataset_name} data: {e}")
                return None

def process_polygon_chunk(args):
    """Process a chunk of polygons with generalized uncertainty analysis"""
    (chunk_polygons, mosaic_files, prefixes, bands_list, statistics, 
     age_classes, age_mosaic_index, age_band,
     trend_mosaic_index, trend_band, 
     uncertainty_pairs_config, basic_stats_config,
     chunk_id, aws_credentials, temp_dir) = args
    
    # Initialize processor
    processor = GeneralizedZonalStats(aws_credentials, temp_dir)
    
    results = []
    
    print(f"🔧 Worker {chunk_id}: Processing {len(chunk_polygons)} polygons")
    
    for idx, (_, polygon_row) in enumerate(chunk_polygons.iterrows()):
        try:
            geometry = polygon_row.geometry
            polygon_crs = chunk_polygons.crs
            result_row = polygon_row.to_dict()
            
            # Extract aligned data for uncertainty analysis
            extracted_data = {}
            
            # Determine which datasets we need
            need_age = age_classes and age_mosaic_index is not None and age_band is not None
            need_trend = trend_mosaic_index is not None and trend_band is not None
            need_uncertainty = uncertainty_pairs_config
            need_basic_stats = basic_stats_config
            
            if need_age or need_trend or need_uncertainty or need_basic_stats:
                
                try:
                    # Build datasets configuration
                    datasets_config = {}
                    
                    # Add age data
                    if need_age and age_mosaic_index < len(mosaic_files):
                        age_raster_paths, age_mosaic_crs = processor.load_tile_index_from_geojson(
                            mosaic_files[age_mosaic_index], geometry, polygon_crs
                        )
                        if age_raster_paths:
                            datasets_config['age'] = {
                                'paths': age_raster_paths,
                                'crs': age_mosaic_crs,
                                'band': age_band
                            }
                    
                    # Add trend data
                    if need_trend and trend_mosaic_index < len(mosaic_files):
                        trend_raster_paths, trend_mosaic_crs = processor.load_tile_index_from_geojson(
                            mosaic_files[trend_mosaic_index], geometry, polygon_crs
                        )
                        if trend_raster_paths:
                            datasets_config['trend'] = {
                                'paths': trend_raster_paths,
                                'crs': trend_mosaic_crs,
                                'band': trend_band
                            }
                    
                    # Add uncertainty pair datasets (mean/std pairs)
                    for pair_name, pair_config in uncertainty_pairs_config.items():
                        for data_type in ['mean', 'std']:
                            mosaic_idx = pair_config[f'{data_type}_mosaic_index']
                            band = pair_config[f'{data_type}_band']
                            
                            if mosaic_idx < len(mosaic_files):
                                raster_paths, mosaic_crs = processor.load_tile_index_from_geojson(
                                    mosaic_files[mosaic_idx], geometry, polygon_crs
                                )
                                if raster_paths:
                                    datasets_config[f'{pair_name}_{data_type}'] = {
                                        'paths': raster_paths,
                                        'crs': mosaic_crs,
                                        'band': band
                                    }
                    
                    # Add basic stats datasets
                    for stat_name, stat_config in basic_stats_config.items():
                        mosaic_idx = stat_config['mosaic_index']
                        band = stat_config['band']
                        
                        if mosaic_idx < len(mosaic_files):
                            raster_paths, mosaic_crs = processor.load_tile_index_from_geojson(
                                mosaic_files[mosaic_idx], geometry, polygon_crs
                            )
                            if raster_paths:
                                datasets_config[stat_name] = {
                                    'paths': raster_paths,
                                    'crs': mosaic_crs,
                                    'band': band
                                }
                    
                    # Extract aligned data
                    if datasets_config:
                        extracted_data = processor.extract_aligned_multi_data(
                            geometry, datasets_config, polygon_crs
                        )
                
                except Exception as e:
                    print(f"⚠️  Warning: Could not extract aligned data: {e}")
            
            # Calculate trend class statistics if we have trend data
            if 'trend' in extracted_data and len(extracted_data['trend']) > 0:
                try:
                    age_data = extracted_data.get('age')
                    trend_class_stats = get_trend_class_stats(
                        extracted_data['trend'], age_data, age_classes
                    )
                    result_row.update(trend_class_stats)
                except Exception as e:
                    print(f"⚠️  Warning: Could not calculate trend class stats: {e}")
            
            # Calculate uncertainty statistics for each mean/std pair
            if uncertainty_pairs_config and extracted_data:
                try:
                    pixel_area_ha = 0.09  # 30m pixels = 0.09 ha
                    
                    for pair_name, pair_config in uncertainty_pairs_config.items():
                        mean_key = f'{pair_name}_mean'
                        std_key = f'{pair_name}_std'
                        
                        if mean_key in extracted_data and std_key in extracted_data:
                            mean_data = extracted_data[mean_key]
                            std_data = extracted_data[std_key]
                            
                            if len(mean_data) > 0 and len(std_data) > 0:
                                print(f"🔧 Worker {chunk_id}: Calculating uncertainty for {pair_name}")
                                
                                # Get age and trend data if available
                                age_data = extracted_data.get('age')
                                trend_data = extracted_data.get('trend')
                                
                                # Calculate uncertainty statistics
                                uncertainty_stats = calculate_uncertainty_stats(
                                    mean_data, std_data, age_data, trend_data, age_classes,
                                    pixel_area_ha=pixel_area_ha, method='monte_carlo'
                                )
                                
                                # Add prefix to distinguish different pairs
                                for key, value in uncertainty_stats.items():
                                    result_row[f"{pair_name}_{key}"] = value
                
                except Exception as e:
                    print(f"⚠️  Warning: Could not calculate uncertainty stats: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate basic statistics for individual datasets
            if basic_stats_config and extracted_data:
                try:
                    for stat_name, stat_config in basic_stats_config.items():
                        if stat_name in extracted_data:
                            data = extracted_data[stat_name]
                            if len(data) > 0:
                                basic_stats = calculate_basic_stats(data, stat_name)
                                result_row.update(basic_stats)
                
                except Exception as e:
                    print(f"⚠️  Warning: Could not calculate basic stats: {e}")
            
            # Process each mosaic for regular statistics (existing pattern)
            for mosaic_idx, (mosaic_file, prefix, bands) in enumerate(zip(mosaic_files, prefixes, bands_list)):
                
                start_time = pd.Timestamp.now()
                
                # Get intersecting raster paths for this polygon and mosaic
                try:
                    raster_paths, mosaic_crs = processor.load_tile_index_from_geojson(
                        mosaic_file, geometry, polygon_crs
                    )
                except Exception as e:
                    print(f"⚠️  Warning: Could not load tiles for mosaic {mosaic_idx}: {e}")
                    raster_paths = []
                
                if not raster_paths:
                    # Add NaN values for this mosaic
                    for stat in statistics:
                        result_row[f"{prefix}{stat}"] = np.nan if stat != 'count' else 0
                    continue
                
                # Calculate basic zonal statistics for this mosaic
                try:
                    # Extract data for each band and calculate basic stats
                    for band in (bands or [1]):
                        band_data = processor.extract_band_data(
                            geometry, raster_paths, mosaic_crs, band
                        )
                        
                        if band_data is not None:
                            valid_data = band_data.compressed()
                            band_suffix = f"_band{band}" if len(bands or [1]) > 1 else ""
                            
                            # Calculate requested statistics
                            if len(valid_data) > 0:
                                for stat in statistics:
                                    if stat == 'mean':
                                        result_row[f"{prefix}{stat}{band_suffix}"] = float(np.mean(valid_data))
                                    elif stat == 'std':
                                        result_row[f"{prefix}{stat}{band_suffix}"] = float(np.std(valid_data))
                                    elif stat == 'count':
                                        result_row[f"{prefix}{stat}{band_suffix}"] = len(valid_data)
                                    elif stat == 'min':
                                        result_row[f"{prefix}{stat}{band_suffix}"] = float(np.min(valid_data))
                                    elif stat == 'max':
                                        result_row[f"{prefix}{stat}{band_suffix}"] = float(np.max(valid_data))
                                    elif stat == 'median':
                                        result_row[f"{prefix}{stat}{band_suffix}"] = float(np.median(valid_data))
                            else:
                                for stat in statistics:
                                    result_row[f"{prefix}{stat}{band_suffix}"] = np.nan if stat != 'count' else 0
                        else:
                            band_suffix = f"_band{band}" if len(bands or [1]) > 1 else ""
                            for stat in statistics:
                                result_row[f"{prefix}{stat}{band_suffix}"] = np.nan if stat != 'count' else 0
                        
                except Exception as e:
                    print(f"⚠️  Warning: Could not calculate stats for mosaic {mosaic_idx}: {e}")
                    # Add NaN values for this mosaic
                    for stat in statistics:
                        result_row[f"{prefix}{stat}"] = np.nan if stat != 'count' else 0
                
                processing_time = (pd.Timestamp.now() - start_time).total_seconds()
                if processing_time > 5:
                    print(f"🔧 Worker {chunk_id}: Mosaic {mosaic_idx} took {processing_time:.2f}s")
            
            results.append(result_row)
            
            if (idx + 1) % 5 == 0:
                print(f"🔧 Worker {chunk_id}: Processed {idx + 1}/{len(chunk_polygons)} polygons")
        
        except Exception as e:
            print(f"❌ Worker {chunk_id}: Error processing polygon {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Worker {chunk_id}: Completed {len(results)} polygons")
    return results

def parse_uncertainty_pairs(uncertainty_config_str):
    """
    Parse uncertainty pair configurations
    Format: 'pair1:mean_mosaic_idx,mean_band,std_mosaic_idx,std_band pair2:...'
    Example: 'biomass:0,1,0,2 carbon:1,1,1,2'
    """
    if not uncertainty_config_str:
        return {}
    
    pairs = {}
    for pair_def in uncertainty_config_str.split():  # Split on whitespace instead of ';'
        if ':' in pair_def:  # Make sure it's a valid pair definition
            pair_name, config = pair_def.split(':')
            mean_mosaic_idx, mean_band, std_mosaic_idx, std_band = map(int, config.split(','))
            
            pairs[pair_name] = {
                'mean_mosaic_index': mean_mosaic_idx,
                'mean_band': mean_band,
                'std_mosaic_index': std_mosaic_idx,
                'std_band': std_band
            }
    
    return pairs

def parse_basic_stats_config(basic_stats_str):
    """
    Parse basic statistics configurations
    Format: 'var1:mosaic_idx,band var2:mosaic_idx,band'
    Example: 'elevation:2,1 slope:2,2'
    """
    if not basic_stats_str:
        return {}
    
    configs = {}
    for stat_def in basic_stats_str.split():  # Split on whitespace instead of ';'
        if ':' in stat_def:  # Make sure it's a valid stat definition
            stat_name, config = stat_def.split(':')
            mosaic_idx, band = map(int, config.split(','))
            
            configs[stat_name] = {
                'mosaic_index': mosaic_idx,
                'band': band
            }
    
    return configs

def parse_age_classes(age_classes_str):
    """Parse age classes from string format like 'young:0-20 mature:21-80 old:81-None'"""
    if not age_classes_str:
        return None
    
    age_classes = {}
    for class_def in age_classes_str.split():  # Split on whitespace instead of ','
        if ':' in class_def:  # Make sure it's a valid class definition
            class_name, age_range = class_def.split(':')
            min_age, max_age = age_range.split('-')
            
            min_age = int(min_age) if min_age != 'None' else None
            max_age = int(max_age) if max_age != 'None' else None
            
            age_classes[class_name] = (min_age, max_age)
    
    return age_classes

def create_uncertainty_trend_age_summary(results_gdf: gpd.GeoDataFrame,
                                     age_classes: Dict,
                                     output_path: str,
                                     polygon_id_col: str = None) -> pd.DataFrame:
    """
    Create summary file with polygon_id, age_class, trend class, and uncertainty statistics
    (mean, std, CV, lower CI, upper CI, count) for all uncertainty pairs
    """
    
    if not age_classes or not HAS_TREND_FUNCTIONS:
        print("Age classes or trend functions not available, skipping trend-age summary")
        return pd.DataFrame()
    
    print("📊 Creating uncertainty summary by trend class and age class...")
    
    # Determine polygon ID column
    if polygon_id_col and polygon_id_col in results_gdf.columns:
        polygon_id_source = polygon_id_col
    elif 'original_index' in results_gdf.columns:
        polygon_id_source = 'original_index'
    else:
        polygon_id_source = results_gdf.index
        results_gdf['temp_polygon_id'] = results_gdf.index
        polygon_id_source = 'temp_polygon_id'
    
    # Define trend categories for summary
    trend_categories = ['strong_decline', 'moderate_decline', 'stable', 'moderate_increase', 'strong_increase']
    
    # Find all uncertainty pair names by looking for 'total_mc_mean' columns
    uncertainty_prefixes = []
    for col in results_gdf.columns:
        if 'total_mc_mean' in col or 'total_analytical_mean' in col:
            # Extract prefix (e.g., "agb_2020_" from "agb_2020_total_mc_mean")
            prefix = col.replace('total_mc_mean', '').replace('total_analytical_mean', '').strip('_')
            if prefix and prefix not in uncertainty_prefixes:
                uncertainty_prefixes.append(prefix)
    
    # Also find any basic stats columns (which have simpler patterns like 'var_mean', 'var_std')
    basic_stats_vars = []
    for col in results_gdf.columns:
        # Look for columns ending with _mean that aren't part of Monte Carlo patterns
        if col.endswith('_mean') and not any(x in col for x in ['mc_', 'analytical_']):
            var_name = col.replace('_mean', '')
            if var_name and var_name not in basic_stats_vars:
                basic_stats_vars.append(var_name)
    
    print(f"Found uncertainty pairs: {uncertainty_prefixes}")
    print(f"Found basic statistics variables: {basic_stats_vars}")
    
    # Determine if we're using monte carlo or analytical method
    using_mc = any('mc_mean' in col for col in results_gdf.columns)
    method_suffix = 'mc' if using_mc else 'analytical'
    
    summary_rows = []
    
    for idx, row in results_gdf.iterrows():
        polygon_id = row[polygon_id_source]
        
        # Process each age class
        for age_class_name, (min_age, max_age) in age_classes.items():
            
            # Get pixel count for this age class
            age_count_col = f"{age_class_name}_{method_suffix}_count"
            n_pixels = row.get(age_count_col, 0)
            if pd.isna(n_pixels):
                n_pixels = 0
            
            # Create base summary row for overall age class
            summary_row = {
                'polygon_id': polygon_id,
                'age_class': age_class_name,
                'age_range': f"{min_age}-{max_age}",
                'trend_class': 'all',
                'n_pixels': int(n_pixels)
            }
            
            # Add uncertainty statistics for this age class across all uncertainty pairs
            for prefix in uncertainty_prefixes:
                # Define column patterns for age class
                stats_cols = {
                    'mean': f"{prefix}_{age_class_name}_{method_suffix}_mean",
                    'std': f"{prefix}_{age_class_name}_{method_suffix}_std",
                    'cv': f"{prefix}_{age_class_name}_{method_suffix}_cv",
                    'lower_ci': f"{prefix}_{age_class_name}_{method_suffix}_lower_ci",
                    'upper_ci': f"{prefix}_{age_class_name}_{method_suffix}_upper_ci",
                    'count': f"{prefix}_{age_class_name}_{method_suffix}_count"
                }
                
                # Add stats columns
                for stat_type, col in stats_cols.items():
                    output_col = f"{prefix}_{stat_type}"
                    stat_val = row.get(col, np.nan if stat_type != 'count' else 0)
                    summary_row[output_col] = stat_val
            
            # Add basic stats variables
            for var in basic_stats_vars:
                # Look for age class specific stats if available
                age_mean_col = f"{var}_{age_class_name}_mean"
                age_std_col = f"{var}_{age_class_name}_std" 
                age_count_col = f"{var}_{age_class_name}_count"
                
                # Fall back to overall stats if age-specific not available
                if age_mean_col not in results_gdf.columns:
                    age_mean_col = f"{var}_mean"
                    age_std_col = f"{var}_std"
                    age_count_col = f"{var}_count"
                
                if age_mean_col in results_gdf.columns:
                    mean_val = row.get(age_mean_col, np.nan)
                    std_val = row.get(age_std_col, np.nan)
                    count_val = row.get(age_count_col, 0)
                    
                    summary_row[f"{var}_mean"] = mean_val
                    summary_row[f"{var}_std"] = std_val
                    summary_row[f"{var}_count"] = count_val
                    
                    # Calculate CV if both mean and std are available
                    if pd.notna(mean_val) and pd.notna(std_val) and abs(mean_val) > 1e-10:
                        summary_row[f"{var}_cv"] = std_val / mean_val
                    else:
                        summary_row[f"{var}_cv"] = np.nan
            
            # Add this all-trends row
            summary_rows.append(summary_row.copy())
            
            # Process trend-specific statistics
            for trend_cat in trend_categories:
                # Create trend-specific row
                trend_row = summary_row.copy()
                trend_row['trend_class'] = trend_cat
                
                # Get pixel count for this trend class
                trend_count_col = f"{age_class_name}_trend_{trend_cat}_count"
                n_trend_pixels = row.get(trend_count_col, 0)
                if pd.isna(n_trend_pixels):
                    n_trend_pixels = 0
                trend_row['n_pixels'] = int(n_trend_pixels)
                
                # Add uncertainty statistics for this trend class
                for prefix in uncertainty_prefixes:
                    # Define column patterns for age-trend combination
                    stats_cols = {
                        'mean': f"{prefix}_{age_class_name}_trend_{trend_cat}_{method_suffix}_mean",
                        'std': f"{prefix}_{age_class_name}_trend_{trend_cat}_{method_suffix}_std",
                        'cv': f"{prefix}_{age_class_name}_trend_{trend_cat}_{method_suffix}_cv",
                        'lower_ci': f"{prefix}_{age_class_name}_trend_{trend_cat}_{method_suffix}_lower_ci",
                        'upper_ci': f"{prefix}_{age_class_name}_trend_{trend_cat}_{method_suffix}_upper_ci",
                        'count': f"{prefix}_{age_class_name}_trend_{trend_cat}_{method_suffix}_count"
                    }
                    
                    # Add stats columns
                    for stat_type, col in stats_cols.items():
                        output_col = f"{prefix}_{stat_type}"
                        stat_val = row.get(col, np.nan if stat_type != 'count' else 0)
                        trend_row[output_col] = stat_val
                
                # Add this trend-specific row
                summary_rows.append(trend_row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    if len(summary_df) > 0:
        # Save summary file
        summary_file = output_path.replace('.gpkg', '_uncertainty_summary.csv').replace('.shp', '_uncertainty_summary.csv')
        if not summary_file.endswith('.csv'):
            summary_file = f"{os.path.splitext(output_path)[0]}_uncertainty_summary.csv"
        
        summary_df.to_csv(summary_file, index=False)
        print(f"📄 Uncertainty trend-age summary saved to: {summary_file}")
        
        # Print some summary statistics
        print(f"📈 Summary contains {len(summary_df)} rows across {summary_df['polygon_id'].nunique()} polygons")
        print(f"📈 Age classes: {', '.join(summary_df['age_class'].unique())}")
        print(f"📈 Trend classes: {', '.join(summary_df['trend_class'].unique())}")
        
        # Show statistics for all uncertainty pairs
        for prefix in uncertainty_prefixes:
            mean_col = f"{prefix}_mean"
            if mean_col in summary_df.columns:
                # Calculate for rows where trend_class='all'
                all_trends_df = summary_df[summary_df['trend_class'] == 'all']
                total_value = all_trends_df[mean_col].sum()
                print(f"📊 {prefix} total: {total_value:.6f}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(
        description="Generalized Zonal Statistics Tool with Monte Carlo Uncertainty Analysis"
    )
    
    # Input/Output
    parser.add_argument('--polygons', required=True, help='Input polygon shapefile/GeoPackage')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--mosaic-geojsons', required=True, 
                       help='Comma-separated list of GeoJSON files containing S3 paths')
    parser.add_argument('--zs-col-prefix', required=True,
                       help='Comma-separated list of column prefixes for each mosaic')
    
    # Band specification
    parser.add_argument('--bands', default='1',
                       help='Comma-separated list of band specifications for each mosaic (e.g., "1,2,1-3")')
    
    # Age data specification
    parser.add_argument('--age-mosaic-index', type=int,
                       help='Index (0-based) of mosaic containing age data (required for age classes)')
    parser.add_argument('--age-band', type=int,
                       help='Band number containing age data (required for age classes)')
    
    # Trend data specification
    parser.add_argument('--trend-mosaic-index', type=int,
                       help='Index (0-based) of mosaic containing trend class data')
    parser.add_argument('--trend-band', type=int,
                       help='Band number containing trend data')
        
    # Uncertainty analysis specification
    parser.add_argument('--uncertainty-pairs', nargs='*',
                       help='Mean/std pairs for uncertainty analysis (format: "name:mean_mosaic,mean_band,std_mosaic,std_band ...")')
    
    # Basic statistics specification
    parser.add_argument('--basic-stats', nargs='*',
                       help='Variables for basic statistics (format: "name:mosaic,band ...")')
    
    # Age classes
    parser.add_argument('--age-classes', nargs='*',
                       help='Age class definitions (e.g., "young:0-20 mature:21-80 old:81-None")')
    
    parser.add_argument('--statistics', default='count,mean,std',
                       help='Comma-separated list of statistics to calculate')
    parser.add_argument('--processes', type=int, default=mp.cpu_count() - 1,
                       help='Number of parallel processes (default: CPU count - 1)')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Number of polygons per chunk')
    
    # Additional options
    parser.add_argument('--filter-query',
                       help='Optional filter query to select polygons (e.g., "area > 100")')
    parser.add_argument('--polygon-id-col',
                       help='Column name containing polygon IDs (for filtering)')
    parser.add_argument('--polygon-ids',
                       help='Comma-separated list of polygon IDs to process (requires --polygon-id-col)')
    parser.add_argument('--polygon-indices',
                       help='Comma-separated list of polygon indices (0-based) to process')
    parser.add_argument('--preserve-index', action='store_true',
                       help='Preserve original dataframe indices in output')
    parser.add_argument('--aws-credentials',
                       help='Path to JSON file with AWS credentials')
    parser.add_argument('--temp-dir',
                       help='Directory for temporary files')
    
    args = parser.parse_args()
    
    # Parse arguments
    mosaic_files = [f.strip() for f in args.mosaic_geojsons.split(',')]
    prefixes = [p.strip() for p in args.zs_col_prefix.split(',')]
    band_specs = [b.strip() for b in args.bands.split(',')]
    statistics = [s.strip() for s in args.statistics.split(',')]
    
    if len(mosaic_files) != len(prefixes):
        raise ValueError("Number of mosaic files must match number of prefixes")
    
    if len(band_specs) == 1:
        band_specs = band_specs * len(mosaic_files)
    elif len(band_specs) != len(mosaic_files):
        raise ValueError("Number of band specifications must be 1 or match number of mosaics")
    
    # Parse age classes
    if args.age_classes:
        age_classes_str = ' '.join(args.age_classes)  # Join the list back into a string
        age_classes = parse_age_classes(age_classes_str)
    else:
        age_classes = None
    
    # Parse uncertainty pairs
    if args.uncertainty_pairs:
        uncertainty_config_str = ' '.join(args.uncertainty_pairs)  # Join the list back into a string
        uncertainty_pairs_config = parse_uncertainty_pairs(uncertainty_config_str)
    else:
        uncertainty_pairs_config = {}
    
    # Parse basic stats configuration
    if args.basic_stats:
        basic_stats_str = ' '.join(args.basic_stats)  # Join the list back into a string
        basic_stats_config = parse_basic_stats_config(basic_stats_str)
    else:
        basic_stats_config = {}
    
    # Parse bands for each mosaic
    processor = GeneralizedZonalStats()
    bands_list = [processor.parse_band_spec(spec) for spec in band_specs]
    
    # Load polygons
    print(f"📍 Loading polygons from {args.polygons}")
    polygons_gdf = gpd.read_file(args.polygons)
    original_count = len(polygons_gdf)
    print(f"   Loaded {original_count} polygons")
    
    # Apply filter if provided
    if args.filter_query:
        try:
            polygons_gdf = polygons_gdf.query(args.filter_query)
            print(f"🔍 Filter applied: {len(polygons_gdf)}/{original_count} polygons selected")
        except Exception as e:
            print(f"⚠️  Warning: Filter query failed: {e}")
    
    if args.polygon_ids and args.polygon_id_col:
        ids = [id.strip() for id in args.polygon_ids.split(',')]
        polygons_gdf = polygons_gdf.loc[polygons_gdf[args.polygon_id_col].isin(ids)]
        print(f"🔍 ID filter: {len(polygons_gdf)} polygons selected using col {args.polygon_id_col}")
    
    if args.polygon_indices:
        indices = [int(i.strip()) for i in args.polygon_indices.split(',')]
        polygons_gdf = polygons_gdf.iloc[indices]
        print(f"🔍 Index filter: {len(polygons_gdf)} polygons selected")
    
    if args.preserve_index:
        polygons_gdf['original_index'] = polygons_gdf.index
    
    # Ensure all polygons are in WGS84 for initial processing
    if polygons_gdf.crs != 'EPSG:4326':
        print(f"🌍 Reprojecting polygons from {polygons_gdf.crs} to EPSG:4326")
        polygons_gdf = polygons_gdf.to_crs('EPSG:4326')
    
    # Create chunks for parallel processing
    chunk_size = args.chunk_size
    chunks = [polygons_gdf.iloc[i:i+chunk_size] for i in range(0, len(polygons_gdf), chunk_size)]
    print(f"🔧 Created {len(chunks)} chunks of max {chunk_size} polygons each")
    
    # Create chunk arguments
    chunk_args = [
        (chunk, mosaic_files, prefixes, bands_list, statistics, 
         age_classes, args.age_mosaic_index, args.age_band,
         args.trend_mosaic_index, args.trend_band,
         uncertainty_pairs_config, basic_stats_config,
         i, args.aws_credentials, args.temp_dir)
        for i, chunk in enumerate(chunks)
    ]
    
    # Process chunks in parallel
    print(f"🚀 Starting parallel processing with {args.processes} processes...")
    
    all_results = []
    if args.processes == 1:
        # Single process for debugging
        for chunk_arg in tqdm(chunk_args, desc="Processing chunks"):
            results = process_polygon_chunk(chunk_arg)
            all_results.extend(results)
    else:
        # Multi-process
        with mp.Pool(processes=args.processes) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_polygon_chunk, chunk_args),
                total=len(chunk_args),
                desc="Processing chunks"
            ))
            
            for results in chunk_results:
                all_results.extend(results)
    
    # Create output GeoDataFrame and save main results
    if all_results:
        print(f"📊 Creating output with {len(all_results)} processed polygons...")
        results_gdf = gpd.GeoDataFrame(all_results, crs=polygons_gdf.crs)
        
        # Save main results
        print(f"💾 Saving results to: {args.output}")
        
        # Determine output format
        if args.output.lower().endswith('.gpkg'):
            results_gdf.to_file(args.output, driver='GPKG', mode='w')
        elif args.output.lower().endswith('.shp'):
            results_gdf.to_file(args.output, driver='ESRI Shapefile')
        elif args.output.lower().endswith(('.geojson', '.json')):
            results_gdf.to_file(args.output, driver='GeoJSON')
        else:
            # Default to GeoPackage
            results_gdf.to_file(args.output, driver='GPKG', mode='w')
        
        # Create summary CSV (without geometry)
        try:
            summary_output = args.output.replace('.gpkg', '_summary.csv').replace('.shp', '_summary.csv').replace('.geojson', '_summary.csv')
            summary_df = results_gdf.drop(columns=['geometry'])
            summary_df.to_csv(summary_output, index=False)
            print(f"📋 Summary CSV saved to: {summary_output}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create summary CSV: {e}")
        
        # Create uncertainty trend-age summary
        try:
            if age_classes and (uncertainty_pairs_config or basic_stats_config):
                create_uncertainty_trend_age_summary(
                    results_gdf, age_classes, args.output, args.polygon_id_col
                )
        except Exception as e:
            print(f"⚠️  Warning: Could not create uncertainty trend-age summary: {e}")
            import traceback
            traceback.print_exc()
        
        print("✅ Processing completed successfully!")
        
        # Create summary CSV (without geometry)
        try:
            summary_output = args.output.replace('.gpkg', '_summary.csv').replace('.shp', '_summary.csv').replace('.geojson', '_summary.csv')
            summary_df = results_gdf.drop(columns=['geometry'])
            summary_df.to_csv(summary_output, index=False)
            print(f"📋 Summary CSV saved to: {summary_output}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create summary CSV: {e}")
        
        # Print summary statistics
        print("\n📈 Summary Statistics:")
        
        # Find numeric columns
        numeric_cols = []
        for col in results_gdf.columns:
            if col != 'geometry' and results_gdf[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        
        # Show stats for first few columns
        for col in numeric_cols[:10]:
            if col in results_gdf.columns:
                if results_gdf[col].notna().any():
                    print(f"   {col}: mean={results_gdf[col].mean():.4g}, non-null={results_gdf[col].notna().sum()}")
                else:
                    print(f"   {col}: all values are null")
                    
    else:
        print("⚠️  No results were produced.")

if __name__ == "__main__":
    main()