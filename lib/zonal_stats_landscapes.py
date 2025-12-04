#!/usr/bin/env python3
"""
Simplified Zonal Statistics Tool using rio_tiler for dynamic mosaic reading
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

warnings.filterwarnings('ignore')

def filter_intersecting_hydrobasins_spatial_index(hydrobasins, spatial_data, buffer_degrees=0.0001, verbose=False):
    """
    Filter hydrobasins to only include those that actually intersect with the spatial data geometries.
    Uses spatial index for better performance.
    
    Parameters:
    -----------
    hydrobasins : geopandas.GeoDataFrame
        Hydrobasins to filter
    spatial_data : str, Path, or geopandas.GeoDataFrame
        Can be:
        - Raster file path (local or s3://) - uses raster extent
        - GeoDataFrame - uses actual geometries
        - Vector file path (shapefile, gpkg, etc.) - uses actual geometries
    buffer_degrees : float
        Buffer around bounds in degrees (only used for raster extent)
    verbose : bool
        Enable verbose output
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Filtered hydrobasins that intersect the spatial data
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box
    from pathlib import Path
    
    if verbose:
        print(f"Original hydrobasins count: {len(hydrobasins)}")
    
    def _identify_spatial_input(data):
        """Identify the type of spatial input."""
        if isinstance(data, gpd.GeoDataFrame):
            return 'geodataframe'
        elif isinstance(data, (str, Path)):
            path_str = str(data).lower()
            # Check for raster extensions
            raster_extensions = ['.tif', '.tiff', '.nc', '.hdf', '.img', '.jp2', '.png', '.jpg']
            vector_extensions = ['.shp', '.gpkg', '.geojson', '.kml', '.gml', '.json']
            
            if any(path_str.endswith(ext) for ext in raster_extensions):
                return 'raster_path'
            elif any(path_str.endswith(ext) for ext in vector_extensions):
                return 'vector_path'
            else:
                # Try to determine by attempting to open
                try:
                    vsi_path = convert_s3_to_vsis3(str(data))
                    with rasterio.open(vsi_path) as src:
                        return 'raster_path'
                except:
                    try:
                        gpd.read_file(str(data))
                        return 'vector_path'
                    except:
                        return 'unknown'
        else:
            return 'unknown'
    
    # Identify input type
    input_type = _identify_spatial_input(spatial_data)
    
    if verbose:
        print(f"Spatial data type detected: {input_type}")
    
    # Get intersection geometry based on input type
    if input_type == 'raster_path':
        # Handle raster file path - create bounding box from raster extent
        vsi_path = convert_s3_to_vsis3(str(spatial_data))
        
        with rasterio.open(vsi_path) as src:
            bounds = src.bounds
            spatial_crs = src.crs
            
            if verbose:
                print(f"Raster bounds: {bounds}")
                print(f"Raster CRS: {spatial_crs}")
        
        # Create bounding box geometry
        if spatial_crs != hydrobasins.crs:
            if verbose:
                print(f"Transforming raster bounds from {spatial_crs} to {hydrobasins.crs}")
            
            from rasterio.warp import transform_bounds
            transformed_bounds = transform_bounds(spatial_crs, hydrobasins.crs, 
                                                bounds.left, bounds.bottom, 
                                                bounds.right, bounds.top)
            intersection_geom = box(transformed_bounds[0] - buffer_degrees,
                                  transformed_bounds[1] - buffer_degrees,
                                  transformed_bounds[2] + buffer_degrees,
                                  transformed_bounds[3] + buffer_degrees)
        else:
            intersection_geom = box(bounds.left - buffer_degrees, 
                                  bounds.bottom - buffer_degrees,
                                  bounds.right + buffer_degrees, 
                                  bounds.top + buffer_degrees)
        
        # Use single geometry for intersection
        test_geometries = [intersection_geom]
    
    elif input_type == 'geodataframe':
        # Handle GeoDataFrame directly - use actual geometries
        spatial_gdf = spatial_data.copy()
        
        if verbose:
            print(f"GeoDataFrame shape: {spatial_gdf.shape}")
            print(f"GeoDataFrame CRS: {spatial_gdf.crs}")
        
        # Transform to hydrobasins CRS if needed
        if spatial_gdf.crs != hydrobasins.crs:
            if verbose:
                print(f"Transforming GeoDataFrame from {spatial_gdf.crs} to {hydrobasins.crs}")
            spatial_gdf = spatial_gdf.to_crs(hydrobasins.crs)
        
        # Use all geometries for intersection testing
        test_geometries = spatial_gdf.geometry.tolist()
    
    elif input_type == 'vector_path':
        # Handle vector file path - use actual geometries
        vsi_path = convert_s3_to_vsis3(str(spatial_data))
        spatial_gdf = gpd.read_file(vsi_path)
        
        if verbose:
            print(f"Vector file shape: {spatial_gdf.shape}")
            print(f"Vector file CRS: {spatial_gdf.crs}")
        
        # Transform to hydrobasins CRS if needed
        if spatial_gdf.crs != hydrobasins.crs:
            if verbose:
                print(f"Transforming vector from {spatial_gdf.crs} to {hydrobasins.crs}")
            spatial_gdf = spatial_gdf.to_crs(hydrobasins.crs)
        
        # Use all geometries for intersection testing
        test_geometries = spatial_gdf.geometry.tolist()
    
    else:
        raise ValueError(f"Unsupported spatial data type: {type(spatial_data)}. "
                        f"Expected raster file path, GeoDataFrame, or vector file path.")
    
    # Create a combined geometry for spatial index querying
    if len(test_geometries) == 1:
        query_geom = test_geometries[0]
    else:
        # For multiple geometries, use the union for spatial indexing
        from shapely.ops import unary_union
        query_geom = unary_union(test_geometries)
    
    # Use spatial index for initial filtering
    sindex = hydrobasins.sindex
    possible_matches_index = list(sindex.intersection(query_geom.bounds))
    possible_matches = hydrobasins.iloc[possible_matches_index]
    
    if verbose:
        print(f"Spatial index candidates: {len(possible_matches)}")
    
    # Perform actual intersection test with all test geometries
    intersecting_indices = []
    
    for idx, basin in possible_matches.iterrows():
        basin_geom = basin.geometry
        
        # Test intersection with any of the test geometries
        intersects = False
        for test_geom in test_geometries:
            if basin_geom.intersects(test_geom):
                intersects = True
                break
        
        if intersects:
            intersecting_indices.append(idx)
    
    # Get the intersecting hydrobasins
    filtered_hydrobasins = hydrobasins.loc[intersecting_indices].copy()
    
    if verbose:
        print(f"Final filtered count: {len(filtered_hydrobasins)}")
        print(f"Reduction: {len(hydrobasins) - len(filtered_hydrobasins)} basins removed")
    
    return filtered_hydrobasins.reset_index(drop=True) #  Now idx will be sequential 0, 1, 2, 3...

def calculate_nmad(data):
    """Calculate Normalized Median Absolute Deviation"""
    if len(data) == 0:
        return np.nan
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    # Normalized MAD (scale factor for normal distribution)
    nmad = 1.4826 * mad
    return nmad

def calculate_mode(data):
    """Calculate mode using scipy.stats"""
    if len(data) == 0:
        return np.nan
    try:
        mode_result = scipy_stats.mode(data, keepdims=False)
        return float(mode_result.mode)
    except:
        # Fallback to most frequent value
        unique, counts = np.unique(data, return_counts=True)
        return float(unique[np.argmax(counts)])

def get_age_class_stats(data, age_classes=None):
    """
    Calculate statistics by age class (similar to carbon_fate.py)
    
    Parameters:
    data: array of values
    age_classes: dict mapping age class ranges to labels
                e.g., {(0, 20): 'young', (21, 40): 'medium', (41, 100): 'old'}
    """
    if age_classes is None:
        # Default age classes (assuming data represents ages or similar)
        age_classes = {
            (0, 10): 'class_0_10',
            (11, 20): 'class_11_20', 
            (21, 40): 'class_21_40',
            (41, 60): 'class_41_60',
            (61, 100): 'class_61_100'
        }
    
    age_stats = {}
    
    for (min_age, max_age), class_name in age_classes.items():
        mask = (data >= min_age) & (data <= max_age)
        class_data = data[mask]
        
        if len(class_data) > 0:
            age_stats[f"{class_name}_count"] = len(class_data)
            age_stats[f"{class_name}_mean"] = float(np.mean(class_data))
            age_stats[f"{class_name}_std"] = float(np.std(class_data))
            age_stats[f"{class_name}_sum"] = float(np.sum(class_data))
        else:
            age_stats[f"{class_name}_count"] = 0
            age_stats[f"{class_name}_mean"] = np.nan
            age_stats[f"{class_name}_std"] = np.nan
            age_stats[f"{class_name}_sum"] = np.nan
    
    return age_stats

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

def get_trend_class_stats(trend_data, age_data=None, age_classes=None):
    """
    Calculate trend class statistics, optionally by age class
    
    Parameters:
    trend_data: array of trend class values (0-10 from Kendall classes)
    age_data: array of age values (optional, for age-stratified analysis)
    age_classes: dict mapping age class ranges to labels
    """
    if not HAS_TREND_FUNCTIONS:
        return {}
    
    # Get trend labels and mapping
    kendall_labels = get_kendall_trend_class_labels()
    trend_mapping = get_trend_category_mapping()
    
    # Remap to simplified trend categories if desired
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
    
    # Overall trend class statistics (original Kendall classes)
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
    #print(f'Age and trend shapes: {age_data.shape}, {trend_data.shape}')
    if age_data is not None and age_classes is not None:
        
        # Handle size mismatch between age and trend data
        if len(age_data) != len(trend_data):
            print(f"Warning: Age data size ({len(age_data)}) != trend data size ({len(trend_data)})")
            print("Skipping age-stratified trend analysis due to size mismatch")
            return trend_stats
        
        # Ensure both arrays are valid (not NaN) at the same locations
        valid_both_mask = valid_trend_mask & ~np.isnan(age_data)
        valid_age_data = age_data[valid_both_mask]
        valid_trend_for_age = trend_data[valid_both_mask]
        valid_simplified_for_age = simplified_trends[valid_both_mask]
        
        for (min_age, max_age), age_class_name in age_classes.items():
            age_mask = (valid_age_data >= min_age) & (valid_age_data <= max_age)
            age_trend_data = valid_trend_for_age[age_mask]
            age_simplified_trends = valid_simplified_for_age[age_mask]
            
            age_total_pixels = len(age_trend_data)
            
            if age_total_pixels > 0:
                # Simplified trend categories by age class
                for category_val, category_name in trend_category_names.items():
                    if category_val == -1:
                        continue
                    count = np.sum(age_simplified_trends == category_val)
                    proportion = count / age_total_pixels if age_total_pixels > 0 else 0
                    
                    trend_stats[f"{age_class_name}_trend_{category_name}_count"] = count
                    trend_stats[f"{age_class_name}_trend_{category_name}_prop"] = proportion
            else:
                # No pixels in this age class - add zero counts
                for category_val, category_name in trend_category_names.items():
                    if category_val == -1:
                        continue
                    trend_stats[f"{age_class_name}_trend_{category_name}_count"] = 0
                    trend_stats[f"{age_class_name}_trend_{category_name}_prop"] = 0.0
    
    return trend_stats

def calculate_biomass_stats(biomass_data, age_data=None, trend_data=None, age_classes=None, pixel_area_ha=0.09):
    """
    Calculate biomass statistics by age and trend classes
    
    Parameters:
    biomass_data: array of biomass density values (Mg/ha)
    age_data: array of age values (optional)
    trend_data: array of trend class values (optional)
    age_classes: dict mapping age class ranges to labels
    pixel_area_ha: area of each pixel in hectares (default 0.09 for 30m pixels)
    """
    
    biomass_stats = {}
    
    # Convert Mg/ha to Pg total for each pixel
    # 1 Mg = 1e-9 Pg, so Mg/ha * ha * 1e-9 = Pg per pixel
    biomass_pg_per_pixel = biomass_data * pixel_area_ha * 1e-9
    
    # Overall biomass statistics
    valid_biomass_mask = ~np.isnan(biomass_data)
    valid_biomass_pg = biomass_pg_per_pixel[valid_biomass_mask]
    
    if len(valid_biomass_pg) > 0:
        biomass_stats['total_biomass_pg'] = float(np.sum(valid_biomass_pg))
        biomass_stats['mean_biomass_density_mgha'] = float(np.mean(biomass_data[valid_biomass_mask]))
        biomass_stats['biomass_pixel_count'] = len(valid_biomass_pg)
    else:
        biomass_stats['total_biomass_pg'] = 0.0
        biomass_stats['mean_biomass_density_mgha'] = np.nan
        biomass_stats['biomass_pixel_count'] = 0
    
    # Age class biomass statistics
    if age_data is not None and age_classes is not None:
        # Ensure arrays are same size
        if len(age_data) != len(biomass_data):
            print(f"Warning: Age data size ({len(age_data)}) != biomass data size ({len(biomass_data)})")
            return biomass_stats
        
        # Create combined validity mask
        valid_both_mask = valid_biomass_mask & ~np.isnan(age_data)
        valid_age_data = age_data[valid_both_mask]
        valid_biomass_for_age = biomass_pg_per_pixel[valid_both_mask]
        valid_density_for_age = biomass_data[valid_both_mask]
        
        for (min_age, max_age), age_class_name in age_classes.items():
            age_mask = (valid_age_data >= min_age) & (valid_age_data <= max_age)
            age_biomass_pg = valid_biomass_for_age[age_mask]
            age_density = valid_density_for_age[age_mask]
            
            if len(age_biomass_pg) > 0:
                biomass_stats[f"{age_class_name}_biomass_pg"] = float(np.sum(age_biomass_pg))
                biomass_stats[f"{age_class_name}_mean_density_mgha"] = float(np.mean(age_density))
                biomass_stats[f"{age_class_name}_biomass_pixels"] = len(age_biomass_pg)
            else:
                biomass_stats[f"{age_class_name}_biomass_pg"] = 0.0
                biomass_stats[f"{age_class_name}_mean_density_mgha"] = np.nan
                biomass_stats[f"{age_class_name}_biomass_pixels"] = 0
    
    # Trend class biomass statistics
    if trend_data is not None and HAS_TREND_FUNCTIONS:
        # Ensure arrays are same size
        if len(trend_data) != len(biomass_data):
            print(f"Warning: Trend data size ({len(trend_data)}) != biomass data size ({len(biomass_data)})")
            return biomass_stats
        
        # Remap trend classes to simplified categories
        simplified_trends = remap_trend_classes(trend_data)
        trend_category_names = {
            0: 'strong_decline', 
            1: 'moderate_decline',
            2: 'stable',
            3: 'moderate_increase', 
            4: 'strong_increase'
        }
        
        # Create combined validity mask
        valid_both_mask = valid_biomass_mask & ~np.isnan(trend_data)
        valid_trend_data = simplified_trends[valid_both_mask]
        valid_biomass_for_trend = biomass_pg_per_pixel[valid_both_mask]
        valid_density_for_trend = biomass_data[valid_both_mask]
        
        for category_val, category_name in trend_category_names.items():
            trend_mask = (valid_trend_data == category_val)
            trend_biomass_pg = valid_biomass_for_trend[trend_mask]
            trend_density = valid_density_for_trend[trend_mask]
            
            if len(trend_biomass_pg) > 0:
                biomass_stats[f"trend_{category_name}_biomass_pg"] = float(np.sum(trend_biomass_pg))
                biomass_stats[f"trend_{category_name}_mean_density_mgha"] = float(np.mean(trend_density))
                biomass_stats[f"trend_{category_name}_biomass_pixels"] = len(trend_biomass_pg)
            else:
                biomass_stats[f"trend_{category_name}_biomass_pg"] = 0.0
                biomass_stats[f"trend_{category_name}_mean_density_mgha"] = np.nan
                biomass_stats[f"trend_{category_name}_biomass_pixels"] = 0
    
    # Age x Trend cross-tabulated biomass statistics
    if (age_data is not None and trend_data is not None and 
        age_classes is not None and HAS_TREND_FUNCTIONS):
        
        # Ensure all arrays are same size
        if len(age_data) == len(trend_data) == len(biomass_data):
            
            simplified_trends = remap_trend_classes(trend_data)
            trend_category_names = {
                0: 'strong_decline', 1: 'moderate_decline', 2: 'stable',
                3: 'moderate_increase', 4: 'strong_increase'
            }
            
            # Create combined validity mask for all three
            valid_all_mask = (valid_biomass_mask & 
                            ~np.isnan(age_data) & 
                            ~np.isnan(trend_data))
            
            valid_age_data = age_data[valid_all_mask]
            valid_trend_data = simplified_trends[valid_all_mask]
            valid_biomass_for_cross = biomass_pg_per_pixel[valid_all_mask]
            valid_density_for_cross = biomass_data[valid_all_mask]
            
            for (min_age, max_age), age_class_name in age_classes.items():
                age_mask = (valid_age_data >= min_age) & (valid_age_data <= max_age)
                
                for category_val, category_name in trend_category_names.items():
                    trend_mask = (valid_trend_data == category_val)
                    combined_mask = age_mask & trend_mask
                    
                    cross_biomass_pg = valid_biomass_for_cross[combined_mask]
                    cross_density = valid_density_for_cross[combined_mask]
                    
                    if len(cross_biomass_pg) > 0:
                        biomass_stats[f"{age_class_name}_trend_{category_name}_biomass_pg"] = float(np.sum(cross_biomass_pg))
                        biomass_stats[f"{age_class_name}_trend_{category_name}_mean_density_mgha"] = float(np.mean(cross_density))
                        biomass_stats[f"{age_class_name}_trend_{category_name}_biomass_pixels"] = len(cross_biomass_pg)
                    else:
                        biomass_stats[f"{age_class_name}_trend_{category_name}_biomass_pg"] = 0.0
                        biomass_stats[f"{age_class_name}_trend_{category_name}_mean_density_mgha"] = np.nan
                        biomass_stats[f"{age_class_name}_trend_{category_name}_biomass_pixels"] = 0
    
    return biomass_stats

def calculate_multi_biomass_stats(biomass_data_dict, age_data=None, trend_data=None, 
                                age_classes=None, pixel_area_ha=0.09):
    """
    Calculate biomass statistics for multiple biomass datasets
    
    Parameters:
    biomass_data_dict: dict with biomass dataset names as keys, arrays as values
    """
    
    all_biomass_stats = {}
    
    for biomass_name, biomass_data in biomass_data_dict.items():
        # Calculate stats for this biomass dataset
        biomass_stats = calculate_biomass_stats(
            biomass_data, age_data, trend_data, age_classes, pixel_area_ha
        )
        
        # Add prefix to all column names
        prefixed_stats = {}
        for key, value in biomass_stats.items():
            if biomass_name == 'primary':  # Keep primary biomass without prefix for compatibility
                prefixed_stats[key] = value
            else:
                prefixed_stats[f"{biomass_name}_{key}"] = value
        
        all_biomass_stats.update(prefixed_stats)
    
    return all_biomass_stats
    
class SimplifiedZonalStats:
    """Simplified zonal statistics processor using rio_tiler"""
    
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
    
    def load_tile_index_from_geojson(self, geojson_file: str, polygon_geometry, polygon_crs) -> Tuple[List[str], CRS]:
        """Load intersecting raster paths from GeoJSON tile index using spatial intersection"""
        
        try:
            # Read the tile index with geopandas
            if geojson_file.startswith('s3://'):
                # For S3 files, use the /vsis3/ path
                from urllib.parse import urlparse
                parsed = urlparse(geojson_file)
                vsi_path = f"/vsis3/{parsed.netloc}{parsed.path}"
                tile_index = gpd.read_file(vsi_path)
            else:
                tile_index = gpd.read_file(geojson_file)
            
            print_str = f"Loaded {len(tile_index)} tiles from {os.path.basename(geojson_file)}"
            
            # Reproject polygon to match tile index CRS if needed
            if polygon_crs != tile_index.crs:
                geom_gdf = gpd.GeoDataFrame([1], geometry=[polygon_geometry], crs=polygon_crs)
                geom_gdf = geom_gdf.to_crs(tile_index.crs)
                polygon_geometry = geom_gdf.geometry.iloc[0]
            
            # Find intersecting tiles
            intersecting_tiles = tile_index[tile_index.intersects(polygon_geometry)]
            print_str += f"; Found {len(intersecting_tiles)} intersecting tiles"
            #print(print_str)
            
            # Extract s3_path from intersecting tiles
            raster_paths = []
            if 's3_path' in intersecting_tiles.columns:
                raster_paths = intersecting_tiles['s3_path'].dropna().tolist()
            elif 'local_path' in intersecting_tiles.columns:
                raster_paths = intersecting_tiles['local_path'].dropna().tolist()
            else:
                print(f"Warning: No 's3_path' or 'local_path' column found")
            
            # Get CRS from first raster file
            mosaic_crs = tile_index.crs  # Default to tile index CRS
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

    def extract_aligned_data(self, geometry, age_paths, age_crs, age_band, 
                            trend_paths, trend_crs, trend_band, polygon_crs):
        """Extract age and trend data on the same spatial grid with the same mask"""
        
        try:
            # Convert geometry to a common CRS (use age CRS as reference)
            geom_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs=polygon_crs)
            if geom_gdf.crs != age_crs:
                geom_gdf = geom_gdf.to_crs(age_crs)
            
            target_geometry = geom_gdf.geometry.iloc[0]
            bounds = target_geometry.bounds
            
            # Calculate dimensions - use consistent resolution for both
            res = 30  # Use same resolution for both datasets
            width = max(10, int((bounds[2] - bounds[0]) / res))
            height = max(10, int((bounds[3] - bounds[1]) / res))
            
            # Limit size
            MAX_SIZE = 1500
            if width > MAX_SIZE or height > MAX_SIZE:
                scale_factor = min(MAX_SIZE/width, MAX_SIZE/height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
            
            # Create transform for this grid
            transform = rasterio.transform.from_bounds(*bounds, width, height)
            
            # Create geometry mask once for both datasets
            from rasterio.features import geometry_mask
            geom_mask = geometry_mask([mapping(target_geometry)], 
                                     out_shape=(height, width),
                                     transform=transform,
                                     invert=True)
            
            # Extract age data
            age_data = None
            with rasterio.env.Env(self.aws_session if self.aws_session else None):
                try:
                    def age_reader(src_path: str, *args, **kwargs):
                        with COGReader(src_path) as cog:
                            return cog.part(bounds, bounds_crs=age_crs, dst_crs=age_crs, 
                                          height=height, width=width, indexes=(age_band,))
                    
                    age_img = mosaic_reader(age_paths, age_reader, pixel_selection=defaults.FirstMethod())
                    
                    if hasattr(age_img, 'array'):
                        age_array = age_img.array[0] if age_img.array.ndim == 3 else age_img.array
                    elif hasattr(age_img, 'data'):
                        age_array = age_img.data[0] if age_img.data.ndim == 3 else age_img.data
                    else:
                        age_array = age_img[0].array[0] if hasattr(age_img[0], 'array') else age_img[0].data[0]
                    
                    # Apply geometry mask and nodata mask
                    age_masked = np.ma.masked_array(age_array, mask=~geom_mask)
                    age_masked = np.ma.masked_invalid(age_masked)
                    
                    # Try to mask nodata values
                    if hasattr(age_img, 'nodata') and age_img.nodata is not None:
                        age_masked = np.ma.masked_equal(age_masked, age_img.nodata)
                    
                    age_data = age_masked
                    
                except Exception as e:
                    print(f"Error extracting age data: {e}")
            
            # Extract trend data on the same grid
            trend_data = None
            if trend_paths:
                # Reproject bounds to trend CRS if needed
                trend_bounds = bounds
                if trend_crs != age_crs:
                    # Transform bounds to trend CRS
                    from rasterio.warp import transform_bounds
                    trend_bounds = transform_bounds(age_crs, trend_crs, *bounds)
                
                with rasterio.env.Env(self.aws_session if self.aws_session else None):
                    try:
                        def trend_reader(src_path: str, *args, **kwargs):
                            with COGReader(src_path) as cog:
                                return cog.part(trend_bounds, bounds_crs=trend_crs, dst_crs=age_crs,  # Reproject to age CRS
                                              height=height, width=width, indexes=(trend_band,))
                        
                        trend_img = mosaic_reader(trend_paths, trend_reader, pixel_selection=defaults.FirstMethod())
                        
                        if hasattr(trend_img, 'array'):
                            trend_array = trend_img.array[0] if trend_img.array.ndim == 3 else trend_img.array
                        elif hasattr(trend_img, 'data'):
                            trend_array = trend_img.data[0] if trend_img.data.ndim == 3 else trend_img.data
                        else:
                            trend_array = trend_img[0].array[0] if hasattr(trend_img[0], 'array') else trend_img[0].data[0]
                        
                        # Apply the SAME geometry mask and nodata mask
                        trend_masked = np.ma.masked_array(trend_array, mask=~geom_mask)
                        trend_masked = np.ma.masked_invalid(trend_masked)
                        
                        # Try to mask nodata values
                        if hasattr(trend_img, 'nodata') and trend_img.nodata is not None:
                            trend_masked = np.ma.masked_equal(trend_masked, trend_img.nodata)
                        
                        trend_data = trend_masked
                        
                    except Exception as e:
                        print(f"Error extracting trend data: {e}")
            
            # Apply common mask - if either is masked, mask both
            if age_data is not None and trend_data is not None:
                # Create combined mask where either dataset is invalid
                combined_mask = age_data.mask | trend_data.mask
                
                # Apply combined mask to both
                age_data = np.ma.masked_array(age_data.data, mask=combined_mask)
                trend_data = np.ma.masked_array(trend_data.data, mask=combined_mask)
                
                # Get final valid data - these will have the same shape!
                age_valid = age_data.compressed()
                trend_valid = trend_data.compressed()
                
                #print(f"Extracted aligned data: age={len(age_valid)}, trend={len(trend_valid)} pixels")
                
                return age_valid, trend_valid
            
            elif age_data is not None:
                return age_data.compressed(), None
            elif trend_data is not None:
                return None, trend_data.compressed()
            else:
                return None, None
                
        except Exception as e:
            print(f"Error in extract_aligned_data: {e}")
            return None, None
            
    def extract_aligned_biomass_data(self, geometry, age_paths, age_crs, age_band, 
                                   trend_paths, trend_crs, trend_band,
                                   biomass_paths, biomass_crs, biomass_band, 
                                   polygon_crs):
        """Extract age, trend, and biomass data on the same spatial grid"""
        
        try:
            # Convert geometry to a common CRS (use age CRS as reference)
            geom_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs=polygon_crs)
            if geom_gdf.crs != age_crs:
                geom_gdf = geom_gdf.to_crs(age_crs)
            
            target_geometry = geom_gdf.geometry.iloc[0]
            bounds = target_geometry.bounds
            
            # Calculate dimensions - use consistent resolution for all datasets
            res = 30  # 30m resolution
            width = max(10, int((bounds[2] - bounds[0]) / res))
            height = max(10, int((bounds[3] - bounds[1]) / res))
            
            # Limit size
            MAX_SIZE = 1500
            if width > MAX_SIZE or height > MAX_SIZE:
                scale_factor = min(MAX_SIZE/width, MAX_SIZE/height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
            
            # Create transform for this grid
            transform = rasterio.transform.from_bounds(*bounds, width, height)
            
            # Create geometry mask once for all datasets
            from rasterio.features import geometry_mask
            geom_mask = geometry_mask([mapping(target_geometry)], 
                                     out_shape=(height, width),
                                     transform=transform,
                                     invert=True)
            
            # Extract age data
            age_data = self._extract_single_dataset(age_paths, age_crs, age_band, bounds, 
                                                   height, width, geom_mask, "age")
            
            # Extract trend data
            trend_data = None
            if trend_paths:
                trend_bounds = bounds
                if trend_crs != age_crs:
                    from rasterio.warp import transform_bounds
                    trend_bounds = transform_bounds(age_crs, trend_crs, *bounds)
                
                trend_data = self._extract_single_dataset(trend_paths, trend_crs, trend_band, 
                                                        trend_bounds, height, width, geom_mask, 
                                                        "trend", target_crs=age_crs)
            
            # Extract biomass data
            biomass_data = None
            if biomass_paths:
                biomass_bounds = bounds
                if biomass_crs != age_crs:
                    from rasterio.warp import transform_bounds
                    biomass_bounds = transform_bounds(age_crs, biomass_crs, *bounds)
                
                biomass_data = self._extract_single_dataset(biomass_paths, biomass_crs, biomass_band,
                                                          biomass_bounds, height, width, geom_mask,
                                                          "biomass", target_crs=age_crs)
            
            # Apply common mask - if any is masked, mask all
            datasets = [data for data in [age_data, trend_data, biomass_data] if data is not None]
            
            if len(datasets) > 1:
                # Create combined mask where any dataset is invalid
                combined_mask = datasets[0].mask.copy()
                for dataset in datasets[1:]:
                    combined_mask = combined_mask | dataset.mask
                
                # Apply combined mask to all datasets
                if age_data is not None:
                    age_data = np.ma.masked_array(age_data.data, mask=combined_mask)
                if trend_data is not None:
                    trend_data = np.ma.masked_array(trend_data.data, mask=combined_mask)
                if biomass_data is not None:
                    biomass_data = np.ma.masked_array(biomass_data.data, mask=combined_mask)
            
            # Get final valid data arrays
            age_valid = age_data.compressed() if age_data is not None else None
            trend_valid = trend_data.compressed() if trend_data is not None else None
            biomass_valid = biomass_data.compressed() if biomass_data is not None else None
            
            print(f"Extracted aligned data: age={len(age_valid) if age_valid is not None else 0}, "
                  f"trend={len(trend_valid) if trend_valid is not None else 0}, "
                  f"biomass={len(biomass_valid) if biomass_valid is not None else 0} pixels")
            
            return age_valid, trend_valid, biomass_valid
            
        except Exception as e:
            print(f"Error in extract_aligned_biomass_data: {e}")
            return None, None, None

    def extract_aligned_multi_biomass_data(self, geometry, age_paths, age_crs, age_band, 
                                     trend_paths, trend_crs, trend_band,
                                     biomass_datasets, polygon_crs):
        """
        Extract age, trend, and multiple biomass datasets on the same spatial grid
        
        Parameters:
        biomass_datasets: list of dicts with keys: 'paths', 'crs', 'band', 'name'
                         Can have multiple entries with same paths/crs but different bands
        """
            
        try:
            # Convert geometry to a common CRS (use age CRS as reference)
            geom_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs=polygon_crs)
            if geom_gdf.crs != age_crs:
                geom_gdf = geom_gdf.to_crs(age_crs)
            
            target_geometry = geom_gdf.geometry.iloc[0]
            bounds = target_geometry.bounds
            
            # Calculate dimensions - use consistent resolution for all datasets
            res = 30  # 30m resolution
            width = max(10, int((bounds[2] - bounds[0]) / res))
            height = max(10, int((bounds[3] - bounds[1]) / res))
            
            # Limit size
            MAX_SIZE = 1500
            if width > MAX_SIZE or height > MAX_SIZE:
                scale_factor = min(MAX_SIZE/width, MAX_SIZE/height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
            
            # Create transform for this grid
            transform = rasterio.transform.from_bounds(*bounds, width, height)
            
            # Create geometry mask once for all datasets
            from rasterio.features import geometry_mask
            geom_mask = geometry_mask([mapping(target_geometry)], 
                                     out_shape=(height, width),
                                     transform=transform,
                                     invert=True)
            
            # Extract age data
            age_data = self._extract_single_dataset(age_paths, age_crs, age_band, bounds, 
                                                   height, width, geom_mask, "age")
            
            # Extract trend data
            trend_data = None
            if trend_paths:
                trend_data = self._extract_single_dataset(
                    trend_paths, trend_crs, trend_band,
                    transform_bounds(age_crs, trend_crs, *bounds) if trend_crs != age_crs else bounds,
                    height, width, geom_mask, "trend", target_crs=age_crs
                )
            
            # Extract multiple biomass datasets - UPDATED to handle same mosaic, different bands
            biomass_data_dict = {}
            
            # Group datasets by mosaic to avoid re-reading same files
            mosaic_groups = {}
            for dataset in biomass_datasets:
                key = (tuple(dataset['paths']), str(dataset['crs']))
                if key not in mosaic_groups:
                    mosaic_groups[key] = []
                mosaic_groups[key].append(dataset)
            
            for (paths_tuple, crs_str), datasets_group in mosaic_groups.items():
                paths = list(paths_tuple)
                crs = datasets_group[0]['crs']  # All in group have same CRS
                
                if not paths:
                    continue
                
                # Calculate bounds for this CRS
                biomass_bounds = bounds
                if crs != age_crs:
                    from rasterio.warp import transform_bounds
                    biomass_bounds = transform_bounds(age_crs, crs, *bounds)
                
                #print(f"📊 Processing mosaic group with {len(datasets_group)} bands from {len(paths)} files")
                
                # Extract data for each band in this mosaic group
                for dataset in datasets_group:
                    biomass_band = dataset['band']
                    biomass_name = dataset['name']
                    
                    biomass_data = self._extract_single_dataset(
                        paths, crs, biomass_band,
                        biomass_bounds, height, width, geom_mask,
                        f"biomass_{biomass_name}", target_crs=age_crs
                    )
                    
                    if biomass_data is not None:
                        biomass_data_dict[biomass_name] = biomass_data
                        #print(f"✅ Extracted {biomass_name} from band {biomass_band}")
            
            # Apply common mask to all datasets
            all_datasets = [age_data, trend_data] + list(biomass_data_dict.values())
            valid_datasets = [d for d in all_datasets if d is not None]
            
            if len(valid_datasets) > 1:
                # Create combined mask
                combined_mask = valid_datasets[0].mask.copy()
                for dataset in valid_datasets[1:]:
                    combined_mask = combined_mask | dataset.mask
                
                # Apply combined mask
                if age_data is not None:
                    age_data = np.ma.masked_array(age_data.data, mask=combined_mask)
                if trend_data is not None:
                    trend_data = np.ma.masked_array(trend_data.data, mask=combined_mask)
                
                for name in biomass_data_dict:
                    biomass_data_dict[name] = np.ma.masked_array(
                        biomass_data_dict[name].data, mask=combined_mask
                    )
            
            # Get compressed arrays
            age_valid = age_data.compressed() if age_data is not None else None
            trend_valid = trend_data.compressed() if trend_data is not None else None
            
            biomass_valid_dict = {}
            for name, data in biomass_data_dict.items():
                biomass_valid_dict[name] = data.compressed()
            
            # print(f"Extracted aligned data: age={len(age_valid) if age_valid is not None else 0}, "
            #       f"trend={len(trend_valid) if trend_valid is not None else 0}, "
            #       f"biomass datasets={list(biomass_valid_dict.keys())}")
            
            return age_valid, trend_valid, biomass_valid_dict
            
        except Exception as e:
            print(f"Error in extract_aligned_multi_biomass_data: {e}")
            return None, None, {}
    
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

    
    def calculate_zonal_stats_rio_tiler(self, geometry, raster_paths: List[str], mosaic_crs: CRS, 
                                      bands: tuple = (1,),
                                      statistics: List[str] = ['mean', 'std', 'min', 'max', 'count'],
                                      age_classes: Dict = None, 
                                      age_data: np.ndarray = None,
                                      extract_trend_data: bool = False) -> Dict[str, float]:
        """Calculate zonal statistics using rio_tiler mosaic functionality with performance optimization"""
        
        try:
            # Convert geometry to the mosaic CRS if needed
            geom_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs='EPSG:4326')
            if geom_gdf.crs != mosaic_crs:
                geom_gdf = geom_gdf.to_crs(mosaic_crs)
            
            target_geometry = geom_gdf.geometry.iloc[0]
            bounds = target_geometry.bounds  # (minx, miny, maxx, maxy)
            
            # Calculate dimensions - optimize for performance
            # Use adaptive resolution based on geometry size
            geom_area = target_geometry.area
            if geom_area > 1000000:  # Large area - use coarser resolution
                res = 60
            elif geom_area > 100000:  # Medium area
                res = 30  
            else:  # Small area - use finer resolution
                res = 15
                
            width = max(10, int((bounds[2] - bounds[0]) / res))  # Minimum 10 pixels
            height = max(10, int((bounds[3] - bounds[1]) / res))
            
            # Prevent excessive memory usage
            MAX_SIZE = 1500
            if width > MAX_SIZE or height > MAX_SIZE:
                scale_factor = min(MAX_SIZE/width, MAX_SIZE/height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
                print(f"Scaled down to {width}x{height} for memory management")
            
            # Optimized reader that only reads the specific band we need
            def optimized_reader(src_path: str, *args, **kwargs):
                try:
                    with COGReader(src_path) as cog:
                        # Only read the specific bands we need - this should be faster'
                        return cog.part(
                            bounds, 
                            bounds_crs=mosaic_crs,
                            dst_crs=mosaic_crs, 
                            height=height, 
                            width=width, 
                            indexes=bands,  # Only read the bands we actually need
                            max_size=None  # Don't let rio_tiler auto-resize
                        )
                except Exception as e:
                    print(f"Error reading {os.path.basename(src_path)}: {e}")
                    return None
            
            # Read data using optimized mosaic approach
            with rasterio.env.Env(self.aws_session if self.aws_session else None):
                try:
                    # Use mosaic_reader with our optimized reader'
                    img_data = mosaic_reader(
                        raster_paths,
                        optimized_reader,
                        pixel_selection=defaults.FirstMethod(),
                        chunk_size=512  # Process in smaller chunks
                    )
                    
                    # Extract the data array
                    if hasattr(img_data, 'array'):
                        data_array = img_data.array
                    elif hasattr(img_data, 'data'):
                        data_array = img_data.data
                    elif isinstance(img_data, tuple):
                        data_array = img_data[0].array if hasattr(img_data[0], 'array') else img_data[0].data
                    else:
                        data_array = img_data
                    
                    # Handle the bands properly
                    if data_array.ndim == 3:
                        if len(bands) == 1:
                            data = data_array[0]  # Single requested band
                        else:
                            # Multiple bands - take first for now
                            data = data_array[0]
                    else:
                        data = data_array
                    
                    # Create transform for the data
                    transform = rasterio.transform.from_bounds(*bounds, width, height)
                    
                    # Create a mask for the geometry
                    from rasterio.features import geometry_mask
                    mask = geometry_mask([mapping(target_geometry)], 
                                       out_shape=data.shape,
                                       transform=transform,
                                       invert=True)
                    #print(f'Data shape: {data.shape}')
                    # Apply geometry mask to data
                    masked_data = np.ma.masked_array(data, mask=~mask)
                    #print(f'Masked data shape: {masked_data.shape}')
                    # Also mask invalid/nodata values
                    masked_data = np.ma.masked_invalid(masked_data)
                    #print(f'Masked data shape: {masked_data.shape}')
                    
                    # Try to get nodata value and mask it
                    try:
                        if hasattr(img_data, 'nodata') and img_data.nodata is not None:
                            masked_data = np.ma.masked_equal(masked_data, img_data.nodata)
                        elif isinstance(img_data, tuple) and hasattr(img_data[0], 'nodata') and img_data[0].nodata is not None:
                            masked_data = np.ma.masked_equal(masked_data, img_data[0].nodata)
                    except:
                        pass
                    
                    # Get valid data
                    valid_data = masked_data.compressed()
                    #valid_data = masked_data
                    #print(f'Valid data shape: {valid_data.shape}')
                    # Handle special cases for raw data extraction
                    if statistics == ['raw_data']:
                        return {'raw_data': valid_data if len(valid_data) > 0 else None}
                    
                    if extract_trend_data:
                        return {'trend_data': valid_data if len(valid_data) > 0 else None}
                    
                    # Calculate basic statistics
                    stats_dict = {}
                    
                    if len(valid_data) > 0:
                        for stat in statistics:
                            if stat == 'mean':
                                stats_dict[stat] = float(np.mean(valid_data))
                            elif stat == 'std':
                                stats_dict[stat] = float(np.std(valid_data))
                            elif stat == 'min':
                                stats_dict[stat] = float(np.min(valid_data))
                            elif stat == 'max':
                                stats_dict[stat] = float(np.max(valid_data))
                            elif stat == 'count':
                                stats_dict[stat] = len(valid_data)
                            elif stat == 'sum':
                                stats_dict[stat] = float(np.sum(valid_data))
                            elif stat == 'median':
                                stats_dict[stat] = float(np.median(valid_data))
                            elif stat == 'mode':
                                stats_dict[stat] = calculate_mode(valid_data)
                            elif stat == 'nmad':
                                stats_dict[stat] = calculate_nmad(valid_data)
                        
                        # Add age class statistics if requested
                        if age_classes:
                            data_for_age_analysis = age_data if age_data is not None else valid_data
                            if data_for_age_analysis is not None and len(data_for_age_analysis) > 0:
                                age_stats = get_age_class_stats(data_for_age_analysis, age_classes)
                                stats_dict.update(age_stats)
                            else:
                                # Add empty age class stats
                                for (min_age, max_age), class_name in age_classes.items():
                                    stats_dict[f"{class_name}_count"] = 0
                                    stats_dict[f"{class_name}_mean"] = np.nan
                                    stats_dict[f"{class_name}_std"] = np.nan
                                    stats_dict[f"{class_name}_sum"] = np.nan
                    else:
                        # No valid data - handle as before
                        for stat in statistics:
                            if stat in ['mode', 'nmad']:
                                stats_dict[stat] = np.nan
                            else:
                                stats_dict[stat] = np.nan if stat != 'count' else 0
                        
                        # Add empty age class stats if requested
                        if age_classes:
                            for (min_age, max_age), class_name in age_classes.items():
                                stats_dict[f"{class_name}_count"] = 0
                                stats_dict[f"{class_name}_mean"] = np.nan
                                stats_dict[f"{class_name}_std"] = np.nan
                                stats_dict[f"{class_name}_sum"] = np.nan
                    
                    return stats_dict
                    
                except Exception as e:
                    print(f"Error reading mosaic data: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return NaN values for all statistics
                    base_stats = {stat: (np.nan if stat != 'count' else 0) for stat in statistics}
                    if age_classes:
                        for (min_age, max_age), class_name in age_classes.items():
                            base_stats[f"{class_name}_count"] = 0
                            base_stats[f"{class_name}_mean"] = np.nan
                            base_stats[f"{class_name}_std"] = np.nan
                            base_stats[f"{class_name}_sum"] = np.nan
                    return base_stats
        
        except Exception as e:
            print(f"Error in zonal statistics calculation: {e}")
            import traceback
            traceback.print_exc()
            base_stats = {stat: (np.nan if stat != 'count' else 0) for stat in statistics}
            if age_classes:
                for (min_age, max_age), class_name in age_classes.items():
                    base_stats[f"{class_name}_count"] = 0
                    base_stats[f"{class_name}_mean"] = np.nan
                    base_stats[f"{class_name}_std"] = np.nan
                    base_stats[f"{class_name}_sum"] = np.nan
            return base_stats

def process_polygon_chunk(args):
    """Process a chunk of polygons with biomass uncertainty calculations"""
    (chunk_polygons, mosaic_files, prefixes, bands_list, statistics, 
     age_classes, age_mosaic_index, age_band, age_biomass_band,
     trend_mosaic_index, trend_band, biomass_mosaic_index, biomass_band,
     additional_biomass_config, chunk_id, aws_credentials, temp_dir) = args
    
    # Initialize processor
    processor = SimplifiedZonalStats(aws_credentials, temp_dir)
    
    results = []
    
    print(f"🔧 Worker {chunk_id}: Processing {len(chunk_polygons)} polygons")
    
    for idx, (_, polygon_row) in enumerate(chunk_polygons.iterrows()):
        try:
            geometry = polygon_row.geometry
            polygon_crs = chunk_polygons.crs
            result_row = polygon_row.to_dict()
            
            # Extract aligned age, trend, and biomass data
            age_data = None
            trend_data = None
            biomass_data_dict = {}
            
            # Determine which datasets we need
            need_age = age_classes and age_mosaic_index is not None and age_band is not None
            need_trend = trend_mosaic_index is not None and trend_band is not None
            need_biomass = (age_biomass_band is not None or 
                          (biomass_mosaic_index is not None and biomass_band is not None) or
                          additional_biomass_config)
            
            if need_age or need_trend or need_biomass:
                try:
                    # Get paths for age data
                    age_raster_paths = []
                    age_mosaic_crs = None
                    if need_age:
                        age_raster_paths, age_mosaic_crs = processor.load_tile_index_from_geojson(
                            mosaic_files[age_mosaic_index], geometry, polygon_crs
                        )
                    
                    # Get paths for trend data  
                    trend_raster_paths = []
                    trend_mosaic_crs = None
                    if need_trend:
                        trend_raster_paths, trend_mosaic_crs = processor.load_tile_index_from_geojson(
                            mosaic_files[trend_mosaic_index], geometry, polygon_crs
                        )
                    
                    # Prepare biomass datasets configuration
                    biomass_datasets = []
                    
                    # Primary biomass from age mosaic band
                    if age_biomass_band is not None and age_raster_paths:
                        biomass_datasets.append({
                            'paths': age_raster_paths,
                            'crs': age_mosaic_crs,
                            'band': age_biomass_band,
                            'name': 'primary'
                        })
                    
                    # Primary biomass from separate mosaic
                    if biomass_mosaic_index is not None and biomass_band is not None:
                        biomass_raster_paths, biomass_mosaic_crs = processor.load_tile_index_from_geojson(
                            mosaic_files[biomass_mosaic_index], geometry, polygon_crs
                        )
                        biomass_datasets.append({
                            'paths': biomass_raster_paths,
                            'crs': biomass_mosaic_crs,
                            'band': biomass_band,
                            'name': 'primary'
                        })
                    
                    # Additional biomass datasets
                    if additional_biomass_config:
                        for config in additional_biomass_config:
                            add_paths, add_crs = processor.load_tile_index_from_geojson(
                                mosaic_files[config['mosaic_index']], geometry, polygon_crs
                            )
                            biomass_datasets.append({
                                'paths': add_paths,
                                'crs': add_crs,
                                'band': config['band'],
                                'name': config['name']
                            })
                    
                    # Extract aligned data with multiple biomass datasets
                    if biomass_datasets or age_raster_paths or trend_raster_paths:
                        age_data, trend_data, biomass_data_dict = processor.extract_aligned_multi_biomass_data(
                            geometry, 
                            age_raster_paths or [], age_mosaic_crs, age_band or 1,
                            trend_raster_paths or [], trend_mosaic_crs, trend_band or 1,
                            biomass_datasets, polygon_crs
                        )
                
                except Exception as e:
                    print(f"⚠️  Warning: Could not extract aligned data: {e}")
            
            # Calculate trend class statistics if we have trend data
            if trend_data is not None and len(trend_data) > 0:
                try:
                    trend_class_stats = get_trend_class_stats(trend_data, age_data, age_classes)
                    result_row.update(trend_class_stats)
                except Exception as e:
                    print(f"⚠️  Warning: Could not calculate trend class stats: {e}")
            
            # Calculate biomass statistics with uncertainty if we have biomass data
            if biomass_data_dict:
                try:
                    # Calculate pixel area in hectares (30m pixels = 0.09 ha)
                    pixel_area_ha = 0.09
                    
                    # Check for mean and std biomass data for uncertainty analysis
                    biomass_keys = list(biomass_data_dict.keys())
                    mean_keys = [k for k in biomass_keys if any(term in k.lower() for term in ['mean', 'avg', 'average'])]
                    std_keys = [k for k in biomass_keys if any(term in k.lower() for term in ['std', 'stdev', 'deviation', 'uncertainty'])]
                    
                    # Group mean and std pairs
                    uncertainty_pairs = []
                    
                    # Try to match mean and std pairs by prefix
                    for mean_key in mean_keys:
                        # Extract prefix (everything before 'mean'/'avg')
                        mean_prefix = mean_key.lower().replace('mean', '').replace('avg', '').replace('average', '').strip('_')
                        
                        # Find matching std key
                        matching_std = None
                        for std_key in std_keys:
                            std_prefix = std_key.lower().replace('std', '').replace('stdev', '').replace('deviation', '').replace('uncertainty', '').strip('_')
                            if std_prefix == mean_prefix or (mean_prefix == '' and std_prefix == ''):
                                matching_std = std_key
                                break
                        
                        if matching_std:
                            uncertainty_pairs.append((mean_key, matching_std, mean_prefix or 'primary'))
                    
                    # Calculate uncertainty statistics for each pair
                    uncertainty_stats_all = {}
                    
                    for mean_key, std_key, dataset_name in uncertainty_pairs:
                        print(f"🔧 Worker {chunk_id}: Calculating uncertainty for {dataset_name} biomass (mean: {mean_key}, std: {std_key})")
                        
                        mean_data = biomass_data_dict[mean_key]
                        std_data = biomass_data_dict[std_key]
                        
                        if len(mean_data) > 0 and len(std_data) > 0:
                            # Calculate uncertainty statistics
                            uncertainty_stats = calculate_biomass_uncertainty_stats(
                                mean_data, std_data, age_data, trend_data, age_classes,
                                pixel_area_ha=pixel_area_ha, method='monte_carlo'
                            )
                            
                            # Add prefix to distinguish different biomass datasets
                            for key, value in uncertainty_stats.items():
                                if dataset_name == 'primary':
                                    uncertainty_stats_all[key] = value
                                else:
                                    uncertainty_stats_all[f"{dataset_name}_{key}"] = value
                    
                    # Add uncertainty stats to result
                    result_row.update(uncertainty_stats_all)
                    
                    if uncertainty_stats_all:
                        print(f"🔧 Worker {chunk_id}: Added {len(uncertainty_stats_all)} uncertainty statistics")
                    
                    # Also calculate standard biomass statistics for individual datasets
                    for name, biomass_data in biomass_data_dict.items():
                        if len(biomass_data) > 0:
                            # Skip if this is part of an uncertainty pair (already processed)
                            if not any(name in [pair[0], pair[1]] for pair in uncertainty_pairs):
                                biomass_stats = calculate_biomass_stats(
                                    biomass_data, age_data, trend_data, age_classes, pixel_area_ha
                                )
                                
                                # Add prefix if not primary
                                if name != 'primary':
                                    prefixed_stats = {}
                                    for key, value in biomass_stats.items():
                                        prefixed_stats[f"{name}_{key}"] = value
                                    result_row.update(prefixed_stats)
                                else:
                                    result_row.update(biomass_stats)
                
                except Exception as e:
                    print(f"⚠️  Warning: Could not calculate biomass stats: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Process each mosaic for regular statistics (existing code)
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
                
                # Calculate statistics for this mosaic
                apply_age_classes = age_classes if (age_mosaic_index is None or mosaic_idx == age_mosaic_index) else None
                
                try:
                    stats = processor.calculate_zonal_stats_rio_tiler(
                        geometry, raster_paths, mosaic_crs, bands, statistics, 
                        apply_age_classes, age_data
                    )
                    
                    # Add stats with prefix
                    for stat_name, stat_value in stats.items():
                        result_row[f"{prefix}{stat_name}"] = stat_value
                        
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

def create_age_class_summary(results_gdf: gpd.GeoDataFrame, 
                           age_mosaic_index: int,
                           age_classes: Dict,
                           prefixes: List[str],
                           statistics: List[str],
                           output_path: str,
                           polygon_id_col: str = None) -> pd.DataFrame:
    """
    Create summary statistics by age class for each polygon
    
    Parameters:
    results_gdf: GeoDataFrame with zonal statistics results
    age_mosaic_index: Index of the mosaic containing age data
    age_classes: Dictionary of age class definitions
    prefixes: List of column prefixes for each mosaic
    statistics: List of statistics that were calculated
    output_path: Base path for output files
    polygon_id_col: Column name to use as polygon ID (if None, uses index)
    """
    
    if not age_classes:
        print("No age classes defined, skipping summary creation")
        return pd.DataFrame()
    
    print("📊 Creating age class summary...")
    
    # Determine polygon ID column
    if polygon_id_col and polygon_id_col in results_gdf.columns:
        polygon_id_source = polygon_id_col
    elif 'original_index' in results_gdf.columns:
        polygon_id_source = 'original_index'
    else:
        polygon_id_source = results_gdf.index
        results_gdf['temp_polygon_id'] = results_gdf.index
        polygon_id_source = 'temp_polygon_id'
    
    summary_rows = []
    
    # Get the prefix for the age mosaic (we'll exclude this from other stats)
    age_prefix = prefixes[age_mosaic_index] if age_mosaic_index is not None else None
    
    # Find age class columns
    age_class_columns = []
    for (min_age, max_age), class_name in age_classes.items():
        age_class_columns.append(class_name)
    
    for idx, row in results_gdf.iterrows():
        polygon_id = row[polygon_id_source]
        
        # Process each age class
        for (min_age, max_age), class_name in age_classes.items():
            
            # Get pixel count for this age class
            count_col = f"{age_prefix}{class_name}_count" if age_prefix else f"{class_name}_count"
            n_pixels = row.get(count_col, 0)
            
            if pd.isna(n_pixels):
                n_pixels = 0
            
            # Create summary row
            summary_row = {
                'polygon_id': polygon_id,
                'age_class': class_name,
                'age_range': f"{min_age}-{max_age}",
                'n_pixels': int(n_pixels)
            }
            
            # Add statistics from each raster (excluding age raster)
            for mosaic_idx, prefix in enumerate(prefixes):
                
                # Skip the age mosaic for additional statistics
                if mosaic_idx == age_mosaic_index:
                    continue
                
                for stat in statistics:
                    if stat == 'raw_data':  # Skip internal statistics
                        continue
                    
                    # For age class specific stats
                    age_stat_col = f"{prefix}{class_name}_{stat}"
                    if age_stat_col in results_gdf.columns:
                        summary_row[f"{prefix}{stat}"] = row.get(age_stat_col, np.nan)
                    else:
                        # If age-specific stat doesn't exist, use overall stat
                        overall_stat_col = f"{prefix}{stat}"
                        if overall_stat_col in results_gdf.columns:
                            summary_row[f"{prefix}{stat}"] = row.get(overall_stat_col, np.nan)
                        else:
                            summary_row[f"{prefix}{stat}"] = np.nan
            
            summary_rows.append(summary_row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    if len(summary_df) > 0:
        # Save summary file
        summary_file = output_path.replace('.gpkg', '_age_class_summary.csv').replace('.shp', '_age_class_summary.csv')
        if not summary_file.endswith('.csv'):
            summary_file = f"{os.path.splitext(output_path)[0]}_age_class_summary.csv"
        
        summary_df.to_csv(summary_file, index=False)
        print(f"📄 Age class summary saved to: {summary_file}")
        
        # Print some summary statistics
        print(f"📈 Summary contains {len(summary_df)} rows across {summary_df['polygon_id'].nunique()} polygons")
        print(f"📈 Age classes: {', '.join(summary_df['age_class'].unique())}")
        
        # Show pixel distribution by age class
        pixel_summary = summary_df.groupby('age_class')['n_pixels'].agg(['sum', 'mean', 'count']).round(2)
        print("📈 Pixel distribution by age class:")
        print(pixel_summary)
    
    return summary_df

def create_polygon_summary(results_gdf: gpd.GeoDataFrame,
                          prefixes: List[str], 
                          statistics: List[str],
                          output_path: str,
                          polygon_id_col: str = None) -> pd.DataFrame:
    """Create a simplified polygon-level summary without geometry"""
    
    print("📊 Creating polygon summary...")
    
    # Determine polygon ID column
    if polygon_id_col and polygon_id_col in results_gdf.columns:
        polygon_id_source = polygon_id_col
    elif 'original_index' in results_gdf.columns:
        polygon_id_source = 'original_index'
    else:
        polygon_id_source = results_gdf.index
        results_gdf['temp_polygon_id'] = results_gdf.index
        polygon_id_source = 'temp_polygon_id'
    
    # Select columns to include
    columns_to_include = [polygon_id_source]
    
    # Add all statistical columns
    for prefix in prefixes:
        for stat in statistics:
            if stat == 'raw_data':
                continue
            col_name = f"{prefix}{stat}"
            if col_name in results_gdf.columns:
                columns_to_include.append(col_name)
    
    # Add any age class columns
    age_class_cols = [col for col in results_gdf.columns if any(
        class_suffix in col for class_suffix in ['_count', '_mean', '_std', '_sum']
        if col.split('_')[-1] in ['count', 'mean', 'std', 'sum']
    )]
    columns_to_include.extend(age_class_cols)
    
    # NEW: Add biomass columns
    biomass_cols = [col for col in results_gdf.columns if any(
        term in col for term in ['biomass_pg', '_density_mgha', 'biomass_pixels']
    )]
    columns_to_include.extend(biomass_cols)
    
    # Remove duplicates and ensure columns exist
    columns_to_include = list(dict.fromkeys(columns_to_include))  # Remove duplicates, preserve order
    columns_to_include = [col for col in columns_to_include if col in results_gdf.columns]
    
    # Create summary dataframe (without geometry)
    summary_df = pd.DataFrame(results_gdf[columns_to_include])
    summary_df.rename(columns={polygon_id_source: 'polygon_id'}, inplace=True)
    
    # Save polygon summary
    polygon_summary_file = output_path.replace('.gpkg', '_polygon_summary.csv').replace('.shp', '_polygon_summary.csv')
    if not polygon_summary_file.endswith('.csv'):
        polygon_summary_file = f"{os.path.splitext(output_path)[0]}_polygon_summary.csv"
    
    summary_df.to_csv(polygon_summary_file, index=False)
    print(f"📄 Polygon summary saved to: {polygon_summary_file}")
    print(f"📈 Polygon summary contains {len(summary_df)} polygons with {len(summary_df.columns)-1} statistical columns")
    
    # NEW: Print biomass summary if available
    biomass_cols = [col for col in summary_df.columns if 'biomass_pg' in col]
    if biomass_cols:
        total_mean_biomass = summary_df['total_mean_biomass_pg'].sum() if 'total_mean_biomass_pg' in summary_df.columns else 0
        total_std_biomass = summary_df['total_std_biomass_pg'].sum() if 'total_std_biomass_pg' in summary_df.columns else 0
        print(f"📊 Total biomass across all polygons (mean +/- std): {total_mean_biomass:.6f} Pg +/- {total_std_biomass:.6f} Pg")
    
    return summary_df

def create_trend_age_summary(results_gdf: gpd.GeoDataFrame,
                           age_classes: Dict,
                           output_path: str,
                           polygon_id_col: str = None) -> pd.DataFrame:
    """
    Create summary file with polygon_id, age_class, n_pixels, trend class counts/proportions,
    and biomass statistics for all biomass datasets
    """
    
    if not age_classes or not HAS_TREND_FUNCTIONS:
        print("Age classes or trend functions not available, skipping trend-age summary")
        return pd.DataFrame()
    
    print("📊 Creating trend class by age class summary...")
    
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
    
    # Find all biomass dataset prefixes (primary has no prefix, others have a prefix_)
    biomass_prefixes = ['']  # For primary biomass (no prefix)
    
    # Find additional biomass datasets by looking for total_biomass_pg columns
    biomass_cols = [col for col in results_gdf.columns if 'total_mean_biomass_pg' in col or 'total_std_biomass_pg' in col]
    for col in biomass_cols:
        if col == 'total_mean_biomass_pg' or col == 'total_std_biomass_pg':  # Skip primary biomass
            continue
        # Extract prefix (e.g., "deadwood_" from "deadwood_total_biomass_pg")
        prefix = col.replace('total_mean_biomass_pg', '').replace('total_std_biomass_pg', '')
        biomass_prefixes.append(prefix)
    
    print(f"Found biomass datasets with prefixes: {biomass_prefixes}")
    
    summary_rows = []
    
    for idx, row in results_gdf.iterrows():
        polygon_id = row[polygon_id_source]
        
        # Process each age class
        for (min_age, max_age), age_class_name in age_classes.items():
            
            # Get pixel count for this age class (from age data)
            age_count_cols = [col for col in results_gdf.columns if col.endswith(f"{age_class_name}_count")]
            n_pixels = 0
            
            if age_count_cols:
                n_pixels = row.get(age_count_cols[0], 0)
                if pd.isna(n_pixels):
                    n_pixels = 0
            
            # Create base summary row
            summary_row = {
                'polygon_id': polygon_id,
                'age_class': age_class_name,
                'age_range': f"{min_age}-{max_age}",
                'n_pixels': int(n_pixels)
            }
            
            # Add trend class counts and proportions for this age class
            for trend_cat in trend_categories:
                count_col = f"{age_class_name}_trend_{trend_cat}_count"
                prop_col = f"{age_class_name}_trend_{trend_cat}_prop"
                
                count_val = row.get(count_col, 0)
                prop_val = row.get(prop_col, 0.0)
                
                if pd.isna(count_val):
                    count_val = 0
                if pd.isna(prop_val):
                    prop_val = 0.0
                
                summary_row[f"{trend_cat}_count"] = int(count_val)
                summary_row[f"{trend_cat}_prop"] = float(prop_val)
            
            # Process biomass data for all datasets
            for prefix in biomass_prefixes:
                # Define common biomass column patterns for age class
                age_biomass_cols = [
                    f"{prefix}{age_class_name}_biomass_pg",
                    f"{prefix}{age_class_name}_mean_density_mgha",
                    f"{prefix}{age_class_name}_biomass_pixels"
                ]
                
                # Add age-level biomass columns for this dataset
                for col in age_biomass_cols:
                    if col in results_gdf.columns:
                        # Remove age class name and prefix to create clean column name
                        if prefix == '':
                            # Primary biomass: remove age_class_name_
                            output_col = col.replace(f"{age_class_name}_", "")
                        else:
                            # Additional biomass: remove prefix and age_class_name_
                            output_col = prefix + col.replace(f"{prefix}{age_class_name}_", "")
                        
                        biomass_val = row.get(col, 0.0 if 'pixels' in col else np.nan)
                        summary_row[output_col] = biomass_val
                
                # Add trend-specific biomass for this dataset
                for trend_cat in trend_categories:
                    trend_biomass_cols = [
                        f"{prefix}{age_class_name}_trend_{trend_cat}_biomass_pg",
                        f"{prefix}{age_class_name}_trend_{trend_cat}_mean_density_mgha", 
                        f"{prefix}{age_class_name}_trend_{trend_cat}_biomass_pixels"
                    ]
                    
                    for col in trend_biomass_cols:
                        if col in results_gdf.columns:
                            # Create clean column name by removing the full pattern
                            # Pattern: {prefix}{age_class_name}_trend_{trend_cat}_{measure}
                            
                            if prefix == '':
                                # Primary biomass: {age_class_name}_trend_{trend_cat}_{measure} -> {trend_cat}_{measure}
                                pattern_to_remove = f"{age_class_name}_trend_{trend_cat}_"
                                output_col = f"{trend_cat}_" + col.replace(pattern_to_remove, "")
                            else:
                                # Additional biomass: {prefix}{age_class_name}_trend_{trend_cat}_{measure} -> {prefix}{trend_cat}_{measure}
                                pattern_to_remove = f"{prefix}{age_class_name}_trend_{trend_cat}_"
                                output_col = f"{prefix}{trend_cat}_" + col.replace(pattern_to_remove, "")
                                
                            biomass_val = row.get(col, 0.0 if 'pixels' in col else np.nan)
                            summary_row[output_col] = biomass_val
            
            summary_rows.append(summary_row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    if len(summary_df) > 0:
        # Save summary file
        summary_file = output_path.replace('.gpkg', '_trend_age_summary.csv').replace('.shp', '_trend_age_summary.csv')
        if not summary_file.endswith('.csv'):
            summary_file = f"{os.path.splitext(output_path)[0]}_trend_age_summary.csv"
        
        summary_df.to_csv(summary_file, index=False)
        print(f"📄 Trend-age summary saved to: {summary_file}")
        
        # Print some summary statistics  
        print(f"📈 Summary contains {len(summary_df)} rows across {summary_df['polygon_id'].nunique()} polygons")
        print(f"📈 Age classes: {', '.join(summary_df['age_class'].unique())}")
        
        # Show column names for debugging
        print(f"📊 Generated columns: {sorted([col for col in summary_df.columns if 'biomass' in col])}")
        
        # Show biomass statistics for all datasets
        for prefix in biomass_prefixes:
            prefix_name = "Primary" if prefix == '' else prefix.rstrip('_').capitalize()
            biomass_col = f"{prefix}biomass_pg" if prefix else "biomass_pg"
            
            if biomass_col in summary_df.columns:
                total_biomass = summary_df[biomass_col].sum()
                print(f"📊 {prefix_name} biomass: {total_biomass:.6f} Pg")
    
    return summary_df

def parse_age_classes(age_class_str):
    """Parse age class string like '0-10:young,11-20:medium,21-40:old' """
    if not age_class_str:
        return None
    
    age_classes = {}
    for class_def in age_class_str.split(','):
        range_part, label = class_def.split(':')
        min_age, max_age = map(int, range_part.split('-'))
        age_classes[(min_age, max_age)] = label.strip()
    
    return age_classes

import numpy as np
from scipy import stats

def estimate_biomass_with_uncertainty_mc(mean_density, std_density, pixel_area_ha=0.09, 
                                        n_simulations=1000, confidence_level=0.95):
    """
    Estimate mean biomass sum and uncertainty using Monte Carlo simulation
    
    Parameters:
    -----------
    mean_density : array
        Mean biomass density values (Mg/ha) for each pixel
    std_density : array  
        Standard deviation of biomass density (Mg/ha) for each pixel
    pixel_area_ha : float
        Area of each pixel in hectares (0.09 ha for 30m pixels)
    n_simulations : int
        Number of Monte Carlo simulations
    confidence_level : float
        Confidence level for uncertainty bounds (e.g., 0.95 for 95% CI)
    """
    
    n_pixels = len(mean_density)
    total_biomass_sims = np.zeros(n_simulations)
    
    # For each simulation
    for sim in range(n_simulations):
        # Sample from normal distribution for each pixel
        # Truncate at zero (no negative biomass)
        pixel_samples = np.maximum(0, 
            np.random.normal(mean_density, std_density))
        
        # Convert to total biomass per pixel (Mg)
        pixel_biomass_mg = pixel_samples * pixel_area_ha
        
        # Sum across all pixels and convert to Pg (1 Mg = 1e-9 Pg)
        total_biomass_pg = np.sum(pixel_biomass_mg) * 1e-9
        
        total_biomass_sims[sim] = total_biomass_pg
    
    # Calculate statistics
    #median_biomass = np.median(total_biomass_sims)
    mean_biomass = np.mean(total_biomass_sims)
    std_biomass = np.std(total_biomass_sims)
    
    # Confidence intervals
    alpha = 1 - confidence_level
    lower_ci = np.percentile(total_biomass_sims, 100 * alpha/2)
    upper_ci = np.percentile(total_biomass_sims, 100 * (1 - alpha/2))
    
    # Additional uncertainty metrics
    cv = std_biomass / mean_biomass  # Coefficient of variation
    iqr = np.percentile(total_biomass_sims, 75) - np.percentile(total_biomass_sims, 25)
    
    return {
        #'median_biomass_pg': median_biomass,
        'mean_biomass_pg': mean_biomass,
        'std_biomass_pg': std_biomass,
        'lower_ci_pg': lower_ci,
        'upper_ci_pg': upper_ci,
        'coefficient_of_variation': cv,
        'interquartile_range_pg': iqr,
        'confidence_level': confidence_level,
        'uncertainty_percent': (upper_ci - lower_ci) / mean_biomass * 100
    }

def estimate_biomass_analytical(mean_density, std_density, pixel_area_ha=0.09):
    """
    Analytical estimation using error propagation (Delta method)
    """
    
    # Convert to biomass per pixel (Mg)
    mean_biomass_per_pixel = mean_density * pixel_area_ha
    std_biomass_per_pixel = std_density * pixel_area_ha
    
    # Sum across pixels
    total_mean_biomass_mg = np.sum(mean_biomass_per_pixel)
    
    # Uncertainty propagation for independent pixels
    # Variance of sum = sum of variances (assuming independence)
    total_variance_mg = np.sum(std_biomass_per_pixel**2)
    total_std_biomass_mg = np.sqrt(total_variance_mg)
    
    # Convert to Pg
    total_mean_biomass_pg = total_mean_biomass_mg * 1e-9
    total_std_biomass_pg = total_std_biomass_mg * 1e-9
    
    # Approximate median (for normal distribution, median ≈ mean)
    # For log-normal, median = exp(log(mean) - 0.5 * (std/mean)^2)
    cv = total_std_biomass_pg / total_mean_biomass_pg
    
    if cv < 0.3:  # Low CV - normal approximation
        median_biomass_pg = total_mean_biomass_pg
    else:  # High CV - log-normal approximation
        # Assume log-normal distribution
        mu_log = np.log(total_mean_biomass_pg) - 0.5 * np.log(1 + cv**2)
        median_biomass_pg = np.exp(mu_log)
    
    # 95% confidence interval (assuming normal)
    ci_95 = 1.96 * total_std_biomass_pg
    
    return {
        #'median_biomass_pg': median_biomass_pg,
        'mean_biomass_pg': total_mean_biomass_pg,
        'std_biomass_pg': total_std_biomass_pg,
        'lower_ci_pg': total_mean_biomass_pg - ci_95,
        'upper_ci_pg': total_mean_biomass_pg + ci_95,
        'coefficient_of_variation': cv,
        'uncertainty_percent': ci_95 / total_mean_biomass_pg * 100
    }

def calculate_biomass_uncertainty_stats(mean_data, std_data, age_data=None, trend_data=None, 
                                      age_classes=None, pixel_area_ha=0.09, method='monte_carlo'):
    """
    Calculate biomass statistics with uncertainty for zonal statistics
    """
    
    biomass_stats = {}
    
    # Overall biomass with uncertainty
    if method == 'monte_carlo':
        overall_stats = estimate_biomass_with_uncertainty_mc(mean_data, std_data, pixel_area_ha)
    else:
        overall_stats = estimate_biomass_analytical(mean_data, std_data, pixel_area_ha)
    
    # Add to results with prefixes
    for key, value in overall_stats.items():
        biomass_stats[f"total_{key}"] = value
    
    # # Age class biomass with uncertainty
    # if age_data is not None and age_classes is not None:
        
    #     if len(age_data) != len(mean_data):
    #         print(f"Warning: Age data size != biomass data size")
    #         return biomass_stats
        
    #     valid_mask = ~np.isnan(mean_data) & ~np.isnan(std_data) & ~np.isnan(age_data)
    #     valid_mean = mean_data[valid_mask]
    #     valid_std = std_data[valid_mask]
    #     valid_age = age_data[valid_mask]
        
    #     for (min_age, max_age), age_class_name in age_classes.items():
    #         age_mask = (valid_age >= min_age) & (valid_age <= max_age)
            
    #         if np.sum(age_mask) > 0:
    #             age_mean = valid_mean[age_mask]
    #             age_std = valid_std[age_mask]
                
    #             if method == 'monte_carlo':
    #                 age_stats = estimate_biomass_with_uncertainty_mc(age_mean, age_std, pixel_area_ha)
    #             else:
    #                 age_stats = estimate_biomass_analytical(age_mean, age_std, pixel_area_ha)
                
    #             # Add age-specific stats
    #             for key, value in age_stats.items():
    #                 biomass_stats[f"{age_class_name}_{key}"] = value
    
    # # Trend class biomass with uncertainty
    # if trend_data is not None:
        
    #     if len(trend_data) != len(mean_data):
    #         print(f"Warning: Trend data size != biomass data size")
    #         return biomass_stats
        
    #     # Remap trend classes
    #     simplified_trends = remap_trend_classes(trend_data)
    #     trend_category_names = {
    #         0: 'strong_decline', 1: 'moderate_decline', 2: 'stable',
    #         3: 'moderate_increase', 4: 'strong_increase'
    #     }
        
    #     valid_mask = (~np.isnan(mean_data) & ~np.isnan(std_data) & 
    #                  ~np.isnan(trend_data))
    #     valid_mean = mean_data[valid_mask]
    #     valid_std = std_data[valid_mask]
    #     valid_trends = simplified_trends[valid_mask]
        
    #     for category_val, category_name in trend_category_names.items():
    #         trend_mask = (valid_trends == category_val)
            
    #         if np.sum(trend_mask) > 0:
    #             trend_mean = valid_mean[trend_mask]
    #             trend_std = valid_std[trend_mask]
                
    #             if method == 'monte_carlo':
    #                 trend_stats = estimate_biomass_with_uncertainty_mc(trend_mean, trend_std, pixel_area_ha)
    #             else:
    #                 trend_stats = estimate_biomass_analytical(trend_mean, trend_std, pixel_area_ha)
                
    #             # Add trend-specific stats
    #             for key, value in trend_stats.items():
    #                 biomass_stats[f"trend_{category_name}_{key}"] = value

    # Age×Trend cross-tabulated biomass with uncertainty
    if (age_data is not None and trend_data is not None and 
        age_classes is not None and HAS_TREND_FUNCTIONS):
        
        # Ensure all arrays are same size
        if len(age_data) == len(trend_data) == len(mean_data) == len(std_data):
            
            simplified_trends = remap_trend_classes(trend_data)
            trend_category_names = {
                0: 'strong_decline', 1: 'moderate_decline', 2: 'stable',
                3: 'moderate_increase', 4: 'strong_increase'
            }
            
            # Create combined validity mask for all four datasets
            valid_all_mask = (~np.isnan(mean_data) & ~np.isnan(std_data) & 
                            ~np.isnan(age_data) & ~np.isnan(trend_data))
            
            valid_age_data = age_data[valid_all_mask]
            valid_trend_data = simplified_trends[valid_all_mask]
            valid_mean_data = mean_data[valid_all_mask]
            valid_std_data = std_data[valid_all_mask]
            
            # For each age×trend combination
            for (min_age, max_age), age_class_name in age_classes.items():
                age_mask = (valid_age_data >= min_age) & (valid_age_data <= max_age)
                
                for category_val, category_name in trend_category_names.items():
                    trend_mask = (valid_trend_data == category_val)
                    combined_mask = age_mask & trend_mask
                    
                    if np.sum(combined_mask) > 0:
                        combo_mean = valid_mean_data[combined_mask]
                        combo_std = valid_std_data[combined_mask]
                        
                        if method == 'monte_carlo':
                            combo_stats = estimate_biomass_with_uncertainty_mc(combo_mean, combo_std, pixel_area_ha)
                        else:
                            combo_stats = estimate_biomass_analytical(combo_mean, combo_std, pixel_area_ha)
                        
                        # Add age×trend-specific stats
                        for key, value in combo_stats.items():
                            biomass_stats[f"{age_class_name}_trend_{category_name}_{key}"] = value
                    else:
                        # No pixels in this age×trend combination - add zero/NaN values
                        if method == 'monte_carlo':
                            zero_stats = {
                                #'median_biomass_pg': 0.0,
                                'mean_biomass_pg': 0.0,
                                'std_biomass_pg': np.nan,
                                'lower_ci_pg': 0.0,
                                'upper_ci_pg': 0.0,
                                'coefficient_of_variation': np.nan,
                                'interquartile_range_pg': np.nan,
                                'uncertainty_percent': np.nan
                            }
                        else:
                            zero_stats = {
                                #'median_biomass_pg': 0.0,
                                'mean_biomass_pg': 0.0,
                                'std_biomass_pg': np.nan,
                                'lower_ci_pg': 0.0,
                                'upper_ci_pg': 0.0,
                                'coefficient_of_variation': np.nan,
                                'uncertainty_percent': np.nan
                            }
                        
                        for key, value in zero_stats.items():
                            biomass_stats[f"{age_class_name}_trend_{category_name}_{key}"] = value

    return biomass_stats
    
def main():
    parser = argparse.ArgumentParser(
        description="Simplified Zonal Statistics Tool with Biomass Analysis"
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
    
    # Trend data specification - NEW
    parser.add_argument('--trend-mosaic-index', type=int,
                       help='Index (0-based) of mosaic containing trend class data')
    parser.add_argument('--trend-band', type=int,
                       help='Band number containing trend class data (Kendall classes 0-10)')
    
    # Processing options
    parser.add_argument('--statistics', default='mean,std,min,max,count',
                       help='Statistics to calculate: mean,std,min,max,count,sum,median,mode,nmad')
    parser.add_argument('--age-classes', 
                       help='Age class definitions like "0-10:young,11-20:medium,21-40:old"')
    parser.add_argument('--processes', type=int, default=4,
                       help='Number of parallel processes')
    parser.add_argument('--chunk-size', type=int, default=50,
                       help='Polygons per chunk')
    
    # AWS/S3 options
    parser.add_argument('--aws-credentials', help='AWS credentials file')
    parser.add_argument('--temp-dir', default='/tmp', help='Temporary directory')
    
    # Polygon selection
    parser.add_argument('--polygon-filter', help='Filter expression for polygons')
    parser.add_argument('--polygon-ids', help='Comma-separated list of polygon IDs')
    parser.add_argument('--polygon-indices', help='Comma-separated list of polygon indices')
    
    # Other options
    parser.add_argument('--preserve-index', action='store_true',
                       help='Preserve original polygon indices')
    parser.add_argument('--polygon-id-col', 
                       help='Column name to use as polygon ID in summary files')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Biomass data specification - UPDATED
    parser.add_argument('--age-biomass-band', type=int,
                       help='Band number in age mosaic containing biomass density (Mg/ha)')
    parser.add_argument('--biomass-mosaic-index', type=int,
                       help='Index (0-based) of separate biomass mosaic')
    parser.add_argument('--biomass-band', type=int,
                       help='Band number in biomass mosaic containing biomass density (Mg/ha)')
    
    # NEW: Additional biomass datasets - UPDATED to handle multiple bands per index
    parser.add_argument('--additional-biomass-indices', type=str,
                       help='Comma-separated list of mosaic indices for additional biomass datasets (e.g., "2,3,2")')
    parser.add_argument('--additional-biomass-bands', type=str,
                       help='Comma-separated list of band numbers for additional biomass datasets (e.g., "1,1,3")')
    parser.add_argument('--additional-biomass-names', type=str,
                       help='Comma-separated list of names for additional biomass datasets (e.g., "secondary,deadwood,tertiary")')
    
    args = parser.parse_args()
    
    # Parse inputs
    mosaic_files = [f.strip() for f in args.mosaic_geojsons.split(',')]
    prefixes = [p.strip() for p in args.zs_col_prefix.split(',')]
    statistics = [s.strip() for s in args.statistics.split(',')]
    age_classes = parse_age_classes(args.age_classes)
    
    # Validate age class requirements
    if age_classes:
        if args.age_mosaic_index is None or args.age_band is None:
            print("❌ Error: --age-mosaic-index and --age-band are required when using --age-classes")
            return
        
        if args.age_mosaic_index >= len(mosaic_files):
            print(f"❌ Error: --age-mosaic-index {args.age_mosaic_index} is out of range (0-{len(mosaic_files)-1})")
            return
        
        print(f"👥 Age data: Mosaic {args.age_mosaic_index} ({mosaic_files[args.age_mosaic_index]}), Band {args.age_band}")
    
    # Validate trend class requirements
    if args.trend_mosaic_index is not None:
        if args.trend_band is None:
            print("❌ Error: --trend-band is required when using --trend-mosaic-index")
            return
        
        if args.trend_mosaic_index >= len(mosaic_files):
            print(f"❌ Error: --trend-mosaic-index {args.trend_mosaic_index} is out of range (0-{len(mosaic_files)-1})")
            return
        
        print(f"📈 Trend data: Mosaic {args.trend_mosaic_index} ({mosaic_files[args.trend_mosaic_index]}), Band {args.trend_band}")

# Parse additional biomass datasets - UPDATED
    additional_biomass_config = []
    if args.additional_biomass_indices and args.additional_biomass_bands:
        indices = [int(x.strip()) for x in args.additional_biomass_indices.split(',')]
        bands = [int(x.strip()) for x in args.additional_biomass_bands.split(',')]
        
        if len(indices) != len(bands):
            print("❌ Error: Number of additional biomass indices must match number of bands")
            return
        
        names = ['secondary', 'tertiary', 'quaternary', 'fifth', 'sixth']  # Extended default names
        if args.additional_biomass_names:
            names = [x.strip() for x in args.additional_biomass_names.split(',')]
        
        for i, (idx, band) in enumerate(zip(indices, bands)):
            if idx >= len(mosaic_files):
                print(f"❌ Error: Additional biomass index {idx} is out of range")
                return
            
            additional_biomass_config.append({
                'mosaic_index': idx,
                'band': band,
                'name': names[i] if i < len(names) else f'biomass_{i+1}'
            })
            
            print(f"🌱 Additional biomass {names[i] if i < len(names) else f'biomass_{i+1}'}: "
                  f"Mosaic {idx} ({mosaic_files[idx]}), Band {band}")
    
    # Parse bands - handle different formats
    band_specs = [b.strip() for b in args.bands.split(',')]
    bands_list = []
    
    for band_spec in band_specs:
        if '-' in band_spec:
            # Handle range like "1-3" -> (1,2,3)
            start, end = map(int, band_spec.split('-'))
            bands = tuple(range(start, end + 1))
        else:
            # Handle single band like "9" -> (9,)
            bands = (int(band_spec),)
        bands_list.append(bands)
    
    # If fewer band specs than mosaics, repeat the last one
    while len(bands_list) < len(mosaic_files):
        bands_list.append(bands_list[-1])
    
    # If more band specs than mosaics, truncate
    bands_list = bands_list[:len(mosaic_files)]
    
    if len(mosaic_files) != len(prefixes):
        print("❌ Error: Number of mosaic files must match number of prefixes")
        return
    
    print(f"📁 Loading polygons from: {args.polygons}")
    print(f"🎯 Band configuration: {list(zip(prefixes, bands_list))}")
    if age_classes:
        print(f"👥 Age classes: {age_classes}")
    
    # Load polygons
    try:
        polygons_gdf = gpd.read_file(args.polygons)
        print(f"📊 Loaded {len(polygons_gdf)} polygons")
    except Exception as e:
        print(f"❌ Error loading polygons: {e}")
        return
    
    # Apply polygon filters
    if args.polygon_filter:
        original_count = len(polygons_gdf)
        polygons_gdf = polygons_gdf.query(args.polygon_filter)
        print(f"🔍 Filter applied: {len(polygons_gdf)}/{original_count} polygons selected")
    
    if args.polygon_ids and args.polygon_id_col:
        ids = [int(id.strip()) for id in args.polygon_ids.split(',')]
        polygons_gdf = polygons_gdf.loc[polygons_gdf[args.polygon_id_col].isin(ids)]
        print(f"🔍 ID filter: {len(polygons_gdf)} polygons selected using col {args.polygon_id_col}")
    
    if args.polygon_indices:
        indices = [int(i.strip()) for i in args.polygon_indices.split(',')]
        polygons_gdf = polygons_gdf.iloc[indices]
        print(f"🔍 Index filter: {len(polygons_gdf)} polygons selected")
    
    if args.preserve_index:
        polygons_gdf['original_index'] = range(len(polygons_gdf))

     # Validate biomass requirements
    if args.age_biomass_band is not None and args.biomass_mosaic_index is not None:
        print("❌ Error: Cannot specify both --age-biomass-band and --biomass-mosaic-index")
        return
    
    if args.biomass_mosaic_index is not None and args.biomass_band is None:
        print("❌ Error: --biomass-band is required when using --biomass-mosaic-index")
        return
    
    if args.biomass_mosaic_index is not None and args.biomass_mosaic_index >= len(mosaic_files):
        print(f"❌ Error: --biomass-mosaic-index {args.biomass_mosaic_index} is out of range")
        return
    
    # Print biomass configuration
    if args.age_biomass_band is not None:
        print(f"🌱 Biomass data: Age mosaic band {args.age_biomass_band}")
    elif args.biomass_mosaic_index is not None:
        print(f"🌱 Biomass data: Mosaic {args.biomass_mosaic_index} ({mosaic_files[args.biomass_mosaic_index]}), Band {args.biomass_band}")
    
    
    # Load mosaic paths and determine CRS
    print("📁 Loading mosaic paths and determining CRS...")
    processor = SimplifiedZonalStats(args.aws_credentials, args.temp_dir)
    
    # try:
    #     print(f'\n\n\nMosaic files: {mosaic_files}\n\n\n')
    #     #mosaic_paths_list, mosaic_crs_list = processor.load_mosaic_paths_and_crs(mosaic_files)
    #     mosaic_paths_list, mosaic_crs_list = processor.load_tile_index_from_geojson(mosaic_files, polygons_gdf.geometry, polygons_gdf.crs)
    # except Exception as e:
    #     print(f"❌ Error loading mosaic paths: {e}")
    #     return
    
    # Ensure all polygons are in WGS84 for initial processing
    if polygons_gdf.crs != 'EPSG:4326':
        print(f"🌍 Reprojecting polygons from {polygons_gdf.crs} to EPSG:4326")
        polygons_gdf = polygons_gdf.to_crs('EPSG:4326')
    
    # Create chunks for parallel processing
    chunk_size = args.chunk_size
    chunks = [polygons_gdf.iloc[i:i+chunk_size] for i in range(0, len(polygons_gdf), chunk_size)]
    print(f"🔧 Created {len(chunks)} chunks of max {chunk_size} polygons each")
    
    # Update chunk args to include biomass parameters
    chunk_args = [
        (chunk, mosaic_files, prefixes, bands_list, statistics, 
         age_classes, args.age_mosaic_index, args.age_band, args.age_biomass_band,
         args.trend_mosaic_index, args.trend_band, 
         args.biomass_mosaic_index, args.biomass_band, additional_biomass_config, i, 
         args.aws_credentials, args.temp_dir)
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
        
        print("✅ Processing completed successfully!")
        
        # Create summary files
        try:
            # Create polygon-level summary (CSV without geometry)
            polygon_summary = create_polygon_summary(
                results_gdf, prefixes, statistics, args.output, 
                polygon_id_col=args.polygon_id_col
            )
            
            # Create age class summary if age classes were used
            if age_classes and args.age_mosaic_index is not None:
                age_summary = create_age_class_summary(
                    results_gdf, args.age_mosaic_index, age_classes, 
                    prefixes, statistics, args.output,
                    polygon_id_col=args.polygon_id_col
                )
            
            # Create trend-age summary if both age and trend data are available
            if (age_classes and args.age_mosaic_index is not None and 
                args.trend_mosaic_index is not None and args.trend_band is not None):
                trend_age_summary = create_trend_age_summary(
                    results_gdf, age_classes, args.output,
                    polygon_id_col=args.polygon_id_col
                )
        except Exception as e:
            print(f"⚠️  Warning: Could not create summary files: {e}")
        
    else:
        print("❌ No results generated")

if __name__ == '__main__':
    main()