#!/usr/bin/env python3
"""
zonal_stats_landscapes.py

Zonal Statistics Extraction Tool for Landscape Analysis - HPC Version with Tile Index

A tool for extracting zonal statistics from rasters using polygon geometries
with parallel processing capabilities, support for polygon index subsets for HPC deployment,
mosaic JSON files with S3 support, and efficient tile index-based spatial queries.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
import warnings
import tempfile
import urllib.parse
from collections import defaultdict
import re

import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.merge import merge
import rasterstats
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import box

# Try to import S3 support
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("Warning: boto3 not available. S3 support disabled.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract zonal statistics from rasters using polygon geometries (HPC version with S3 and tile index support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with tile index
  %(prog)s -p polygons.gpkg -m mosaic.json -s mean,std,count -o output.gpkg
  
  # HPC usage with custom tile index
  %(prog)s -p polygons.gpkg -m mosaic.json -s mean,std -o output.gpkg --tindex custom_tiles.gpkg --indices 0,1,2,3,4
        """
    )
    
    # Required arguments (except when generating index files)
    parser.add_argument(
        '-p', '--polygons',
        required=True,
        type=str,
        help='Path to input geopackage containing polygons'
    )
    
    # Index selection options (mutually exclusive)
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument(
        '--indices',
        type=str,
        help='Comma-separated list of polygon indices to process (0-based)'
    )
    index_group.add_argument(
        '--index-range',
        type=str,
        help='Range of indices to process (e.g., "100-199" for indices 100 to 199 inclusive)'
    )
    index_group.add_argument(
        '--index-file',
        type=str,
        help='Path to file containing polygon indices to process (one per line)'
    )
    
    # HPC utility options
    parser.add_argument(
        '--generate-index-files',
        action='store_true',
        help='Generate index files for HPC batch processing'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of polygons per batch when generating index files (default: 1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for generated index files (default: current directory)'
    )
    
    # Show polygon count and exit
    parser.add_argument(
        '--count-polygons',
        action='store_true',
        help='Show total polygon count and exit'
    )
    
    # Raster input (mutually exclusive with mosaic)
    raster_group = parser.add_mutually_exclusive_group(required=False)
    raster_group.add_argument(
        '-r', '--rasters',
        type=str,
        help='Comma-separated list of raster file paths, or single raster path'
    )
    raster_group.add_argument(
        '-m', '--mosaic',
        type=str,
        help='Path to mosaic JSON file representing tiled rasters (supports S3 paths)'
    )
    
    # Tile index
    parser.add_argument(
        '--tindex',
        type=str,
        default='s3://maap-ops-workspace/shared/montesano/databank/boreal_tiles_v004_model_ready.gpkg',
        help='Path to tile index geopackage with tile_num column (default: s3://maap-ops-workspace/shared/montesano/databank/boreal_tiles_v004_model_ready.gpkg)'
    )
    
    # Statistics and output
    parser.add_argument(
        '-s', '--statistics',
        type=str,
        help='Comma-separated list of statistics (e.g., mean,std,count,min,max,median)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output geopackage'
    )
    
    # Optional arguments
    parser.add_argument(
        '-c', '--columns',
        type=str,
        default='',
        help='Comma-separated list of columns to include in output (default: all columns)'
    )
    
    parser.add_argument(
        '--processes',
        type=int,
        default=mp.cpu_count(),
        help=f'Number of processes for parallel execution (default: {mp.cpu_count()})'
    )
    
    parser.add_argument(
        '--nodata',
        type=float,
        help='NoData value to use for raster processing'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='zs_',
        help='Prefix for statistics columns (default: "zs_")'
    )

    parser.add_argument(
        '--zs-col-prefix',
        type=str,
        help='Comma-separated list of custom prefixes for statistics columns, one for each raster/mosaic file (e.g., "AGB_H30_2024,AGB_H30_2025"). If not provided, uses --prefix with _m1, _m2, etc. suffixes'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of polygons to process per chunk (default: 100)'
    )
    
    parser.add_argument(
        '--preserve-index',
        action='store_true',
        help='Preserve original polygon indices in output'
    )
    
    parser.add_argument(
        '--temp-dir',
        type=str,
        help='Directory for temporary files (default: system temp)'
    )
    
    parser.add_argument(
        '--s3-profile',
        type=str,
        help='AWS profile name for S3 access'
    )
    
    parser.add_argument(
        '--mosaic-strategy',
        type=str,
        choices=['tindex', 'first', 'all'],
        default='tindex',
        help='Strategy for selecting tiles: tindex (use tile index for spatial query), first (first tile only), all (all tiles)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()

def calculate_nmad(values):
    """
    Calculate Normalized Median Absolute Deviation (NMAD).
    
    NMAD = 1.4826 * median(|xi - median(x)|)
    
    The factor 1.4826 makes NMAD a consistent estimator for the standard deviation
    of a normal distribution.
    """
    values = np.array(values)
    # Remove NaN values
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return np.nan
    
    median_val = np.median(values)
    mad = np.median(np.abs(values - median_val))
    nmad = 1.4826 * mad
    
    return nmad

def calculate_zonal_stats_with_custom(geometry, raster_path, statistics, nodata_value):
    """
    Calculate zonal statistics including custom functions like NMAD and percentiles.
    """
    # Separate built-in stats from custom stats
    builtin_stats = []
    custom_stats = []
    
    for stat in statistics:
        if stat == 'nmad' or stat.startswith('p'):  # p10, p25, p50, p90, etc.
            custom_stats.append(stat)
        else:
            builtin_stats.append(stat)
    
    # Calculate built-in statistics
    if builtin_stats:
        builtin_results = rasterstats.zonal_stats(
            [geometry],
            raster_path,
            stats=builtin_stats,
            nodata=nodata_value,
            all_touched=True
        )[0]
    else:
        builtin_results = {}
    
    # Calculate custom statistics if needed
    custom_results = {}
    if custom_stats:
        try:
            # Extract pixel values for custom calculations
            pixel_values = rasterstats.zonal_stats(
                [geometry],
                raster_path,
                stats=[],  # No built-in stats
                raster_out=True,  # Return pixel values
                nodata=nodata_value,
                all_touched=True
            )[0]
            
            if pixel_values and 'mini_raster_array' in pixel_values:
                # Get the pixel values
                values = pixel_values['mini_raster_array'].compressed()  # Removes masked/nodata values
                
                # Calculate custom statistics
                for stat in custom_stats:
                    if stat == 'nmad':
                        custom_results['nmad'] = calculate_nmad(values)
                    elif stat.startswith('p') and stat[1:].isdigit():
                        # Handle percentiles: p10, p25, p50, p90, etc.
                        percentile = int(stat[1:])
                        if 0 <= percentile <= 100:
                            custom_results[stat] = np.percentile(values, percentile) if len(values) > 0 else np.nan
                        else:
                            custom_results[stat] = np.nan
            else:
                # No pixel values available
                for stat in custom_stats:
                    custom_results[stat] = np.nan
                    
        except Exception as e:
            # Error in custom calculation
            print(f"Warning: Error calculating custom statistics: {e}")
            for stat in custom_stats:
                custom_results[stat] = np.nan
    
    # Combine results
    combined_results = {**builtin_results, **custom_results}
    
    return combined_results

# def calculate_zonal_stats_with_custom(geometry, raster_path, statistics, nodata_value):
#     """
#     Calculate zonal statistics including custom functions like NMAD.
#     """
#     # Separate built-in stats from custom stats
#     builtin_stats = [stat for stat in statistics if stat != 'nmad']
#     custom_stats = [stat for stat in statistics if stat == 'nmad']
    
#     # Calculate built-in statistics
#     if builtin_stats:
#         builtin_results = rasterstats.zonal_stats(
#             [geometry],
#             raster_path,
#             stats=builtin_stats,
#             nodata=nodata_value,
#             all_touched=True
#         )[0]
#     else:
#         builtin_results = {}
    
#     # Calculate custom statistics if needed
#     custom_results = {}
#     if custom_stats:
#         # Extract pixel values for custom calculations
#         pixel_values = rasterstats.zonal_stats(
#             [geometry],
#             raster_path,
#             stats=[],  # No built-in stats
#             raster_out=True,  # Return pixel values
#             nodata=nodata_value,
#             all_touched=True
#         )[0]
        
#         if pixel_values and 'mini_raster_array' in pixel_values:
#             # Get the pixel values
#             values = pixel_values['mini_raster_array'].compressed()  # Removes masked/nodata values
            
#             # Calculate custom statistics
#             for stat in custom_stats:
#                 if stat == 'nmad':
#                     custom_results['nmad'] = calculate_nmad(values)
    
#     # Combine results
#     combined_results = {**builtin_results, **custom_results}
    
#     return combined_results
    
class S3FileManager:
    """Manage S3 file operations."""
    
    def __init__(self, profile_name=None, temp_dir=None):
        if not S3_AVAILABLE:
            raise ImportError("boto3 is required for S3 support")
        
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.local_files = {}  # Cache for downloaded files
        
        # Initialize S3 client
        try:
            if profile_name:
                session = boto3.Session(profile_name=profile_name)
                self.s3_client = session.client('s3')
            else:
                self.s3_client = boto3.client('s3')
        except Exception as e:
            raise RuntimeError(f"Failed to initialize S3 client: {e}")
    
    @staticmethod
    def is_s3_path(path):
        """Check if path is an S3 URI - static method for easy access."""
        return isinstance(path, str) and path.startswith('s3://')
    
    def parse_s3_path(self, s3_path):
        """Parse S3 path into bucket and key."""
        if not self.is_s3_path(s3_path):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        # Remove s3:// and split
        path_parts = s3_path[5:].split('/', 1)
        if len(path_parts) != 2:
            raise ValueError(f"Invalid S3 path format: {s3_path}")
        
        bucket, key = path_parts
        return bucket, key
    
    def download_file(self, s3_path, force_download=False):
        """Download S3 file to local temp directory."""
        if not self.is_s3_path(s3_path):
            return s3_path  # Return as-is if not S3 path
        
        # Check if already downloaded
        if s3_path in self.local_files and not force_download:
            local_path = self.local_files[s3_path]
            if os.path.exists(local_path):
                return local_path
        
        try:
            bucket, key = self.parse_s3_path(s3_path)
            
            # Create local filename
            filename = os.path.basename(key)
            local_path = os.path.join(self.temp_dir, f"s3_{hash(s3_path) % 100000}_{filename}")
            
            # Download file
            self.s3_client.download_file(bucket, key, local_path)
            self.local_files[s3_path] = local_path
            
            return local_path
            
        except ClientError as e:
            raise RuntimeError(f"Failed to download {s3_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error downloading {s3_path}: {e}")
    
    def cleanup(self):
        """Clean up downloaded temporary files."""
        for s3_path, local_path in self.local_files.items():
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {local_path}: {e}")
        
        self.local_files.clear()


class TileIndexManager:
    """Manage tile index operations for efficient spatial queries."""
    
    def __init__(self, tindex_path, s3_manager=None):
        self.tindex_path = tindex_path
        self.s3_manager = s3_manager
        self.tile_gdf = None
        self.load_tile_index()
    
    def load_tile_index(self):
        """Load the tile index geopackage."""
        print(f"🔧 Loading tile index: {self.tindex_path}")
        
        try:
            if self.tindex_path.startswith('s3://'):
                if self.s3_manager:
                    # Download S3 file to local temp
                    local_tindex = self.s3_manager.download_file(self.tindex_path)
                    self.tile_gdf = gpd.read_file(local_tindex)
                else:
                    # Try VSI S3 access
                    s3_vsi_path = f"/vsis3/{self.tindex_path[5:]}"
                    self.tile_gdf = gpd.read_file(s3_vsi_path)
            else:
                # Local file
                self.tile_gdf = gpd.read_file(self.tindex_path)
            
            print(f"✅ Loaded tile index: {len(self.tile_gdf)} tiles")
            print(f"   CRS: {self.tile_gdf.crs}")
            print(f"   Columns: {list(self.tile_gdf.columns)}")
            
            # Verify tile_num column exists
            if 'tile_num' not in self.tile_gdf.columns:
                raise ValueError("Tile index must contain 'tile_num' column")
            
            # Show sample tile numbers
            sample_tile_nums = self.tile_gdf['tile_num'].head().tolist()
            print(f"   Sample tile numbers: {sample_tile_nums}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tile index from {self.tindex_path}: {e}")
    
    def get_intersecting_tile_numbers(self, geometry):
        """Get tile numbers that intersect with the geometry."""
        print(f"🔧 Finding tiles that intersect with polygon...")
        print(f"   Polygon bounds: {geometry.bounds}")
        
        try:
            # Create GeoDataFrame from single geometry for spatial operations
            geom_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs=self.tile_gdf.crs)
            
            # Perform spatial intersection
            intersecting_tiles = gpd.overlay(self.tile_gdf, geom_gdf, how='intersection')
            
            if len(intersecting_tiles) == 0:
                print("⚠️  No intersecting tiles found using overlay")
                
                # Fallback: try sjoin (spatial join)
                print("   Trying spatial join as fallback...")
                intersecting_tiles = gpd.sjoin(self.tile_gdf, geom_gdf, how='inner', predicate='intersects')
            
            tile_numbers = intersecting_tiles['tile_num'].unique().tolist()
            
            print(f"✅ Found {len(tile_numbers)} intersecting tiles: {tile_numbers}")
            
            return tile_numbers
            
        except Exception as e:
            print(f"❌ Error finding intersecting tiles: {e}")
            
            # Emergency fallback: use bounding box intersection
            print("   Using bounding box intersection as emergency fallback...")
            return self._bbox_intersection_fallback(geometry)
    
    def _bbox_intersection_fallback(self, geometry):
        """Fallback method using bounding box intersection."""
        poly_bounds = geometry.bounds
        poly_bbox = box(poly_bounds[0], poly_bounds[1], poly_bounds[2], poly_bounds[3])
        
        intersecting_tile_nums = []
        
        for idx, tile_row in self.tile_gdf.iterrows():
            tile_geom = tile_row.geometry
            if poly_bbox.intersects(tile_geom):
                intersecting_tile_nums.append(tile_row['tile_num'])
        
        print(f"   Fallback found {len(intersecting_tile_nums)} tiles: {intersecting_tile_nums}")
        return intersecting_tile_nums


class MosaicHandler:
    """Handle mosaic JSON operations with tile index efficiency."""
    
    def __init__(self, mosaic_path, s3_manager=None, tile_index_manager=None):
        self.mosaic_path = mosaic_path
        self.s3_manager = s3_manager
        self.tile_index_manager = tile_index_manager
        
        # Load configuration (now handles S3 paths)
        self.config = self._load_mosaic_config()
        
        # Extract bounds if available
        self.bounds = self.config.get('bounds')
        
        # Determine CRS from first raster file
        self.crs = self._determine_crs_from_first_raster()
        
        #print(f"🔧 Mosaic initialized with CRS: {self.crs}")
        print(f"🔧 Total tiles in mosaic: {len(self.config.get('tiles', {}))}")
    
    def _load_mosaic_config(self):
        """Load mosaic configuration from local or S3 path."""
        try:
            if self.mosaic_path.startswith('s3://'):
                print(f"🔧 Loading mosaic JSON from S3: {self.mosaic_path}")
                
                if self.s3_manager:
                    # Download S3 file to local temp
                    local_mosaic_path = self.s3_manager.download_file(self.mosaic_path)
                    print(f"✅ Downloaded mosaic JSON to: {local_mosaic_path}")
                    with open(local_mosaic_path, 'r') as f:
                        config = json.load(f)
                else:
                    # No S3 manager available
                    raise RuntimeError(f"S3 manager required to access mosaic JSON at {self.mosaic_path}")
            else:
                # Local file
                print(f"🔧 Loading mosaic JSON from local path: {self.mosaic_path}")
                with open(self.mosaic_path, 'r') as f:
                    config = json.load(f)
            
            print(f"✅ Mosaic JSON loaded successfully")
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load mosaic config from {self.mosaic_path}: {e}")
    
    def get_first_raster_file(self):
        """Get the first raster file from the mosaic - ONLY for CRS detection."""
        if not self.config or 'tiles' not in self.config:
            return None
            
        tiles = self.config['tiles']
        
        for quadkey, file_list in tiles.items():
            if file_list and isinstance(file_list, list) and len(file_list) > 0:
                return file_list[0]
        
        return None
    
    def _is_s3_path(self, path):
        """Simple check if path is S3."""
        return isinstance(path, str) and path.startswith('s3://')
    
    def _get_default_crs(self):
        """Get default CRS - use the Albers projection."""
        albers_wkt = '''PROJCS["unnamed",
            GEOGCS["GRS 1980(IUGG, 1980)",
                DATUM["unknown",
                    SPHEROID["GRS80",6378137,298.257222101],
                    TOWGS84[0,0,0,0,0,0,0]],
                PRIMEM["Greenwich",0],
                UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],
            PROJECTION["Albers_Conic_Equal_Area"],
            PARAMETER["latitude_of_center",40],
            PARAMETER["longitude_of_center",180],
            PARAMETER["standard_parallel_1",50],
            PARAMETER["standard_parallel_2",70],
            PARAMETER["false_easting",0],
            PARAMETER["false_northing",0],
            UNIT["metre",1,AUTHORITY["EPSG","9001"]],
            AXIS["Easting",EAST],
            AXIS["Northing",NORTH]]'''
        
        return CRS.from_wkt(albers_wkt)
    
    def _determine_crs_from_first_raster(self):
        """Determine CRS by reading the first raster file - ONLY for CRS detection."""
        first_file = self.get_first_raster_file()
        
        if not first_file:
            print("❌ No first raster file found for CRS detection")
            return self._get_default_crs()
        
        print(f"🔍 Getting CRS from first file: {os.path.basename(first_file)}")
        
        try:
            if self._is_s3_path(first_file):
                s3_vsi_path = f"/vsis3/{first_file[5:]}"
                with rasterio.open(s3_vsi_path) as src:
                    crs = src.crs
                    #print(f"✅ CRS from first file: {crs}")
                    return crs
            else:
                if os.path.exists(first_file):
                    with rasterio.open(first_file) as src:
                        crs = src.crs
                        #print(f"✅ CRS from first file: {crs}")
                        return crs
        except Exception as e:
            print(f"❌ Could not get CRS from first file: {e}")
        
        default_crs = self._get_default_crs()
        print(f"⚠️  Using default CRS: {default_crs}")
        return default_crs
    
    def _extract_tile_num_from_filename(self, filename):
        """Extract tile number from filename (7-digit zero-padded number before .tif)."""
        # Pattern: *_0003567.tif -> tile_num = 3567
        # Look for pattern: underscore + 7 digits + .tif at end
        pattern = r'_(\d{7})\.tif$'
        match = re.search(pattern, filename)
        
        if match:
            # Convert to int to remove leading zeros
            tile_num = int(match.group(1))
            return tile_num
        
        return None
    
    def find_raster_files_by_tile_numbers(self, tile_numbers):
        """Find raster files that match the given tile numbers."""
        print(f"🔧 Finding raster files for {len(tile_numbers)} tile numbers: {tile_numbers}")
        
        matching_files = []
        tiles_config = self.config.get('tiles', {})
        
        # Create a mapping of tile_num to files
        tile_num_to_files = {}
        
        for quadkey, file_list in tiles_config.items():
            if file_list and isinstance(file_list, list):
                for raster_file in file_list:
                    tile_num = self._extract_tile_num_from_filename(raster_file)
                    if tile_num is not None:
                        if tile_num not in tile_num_to_files:
                            tile_num_to_files[tile_num] = []
                        tile_num_to_files[tile_num].append(raster_file)
        
        print(f"🔧 Built mapping for {len(tile_num_to_files)} tile numbers from mosaic")
        
        # Find files for requested tile numbers
        found_tile_nums = []
        for tile_num in tile_numbers:
            if tile_num in tile_num_to_files:
                matching_files.extend(tile_num_to_files[tile_num])
                found_tile_nums.append(tile_num)
                print(f"   ✅ Tile {tile_num:07d}: found {len(tile_num_to_files[tile_num])} files")
            else:
                print(f"   ⚠️  Tile {tile_num:07d}: no matching files found")
        
        print(f"✅ Found files for {len(found_tile_nums)}/{len(tile_numbers)} requested tiles")
        print(f"✅ Total matching files: {len(matching_files)}")
        
        return matching_files
    
    def get_intersecting_tiles_from_index(self, geometry):
        """Get raster files using tile index intersection."""
        if not self.tile_index_manager:
            print("❌ No tile index manager available")
            return []
        
        # Get intersecting tile numbers
        tile_numbers = self.tile_index_manager.get_intersecting_tile_numbers(geometry)
        
        if not tile_numbers:
            print("❌ No intersecting tile numbers found")
            return []
        
        # Find corresponding raster files
        raster_files = self.find_raster_files_by_tile_numbers(tile_numbers)
        
        return raster_files
    
    def get_intersecting_tiles(self, geometry, strategy='tindex'):
        """Get raster files that intersect with the given geometry."""
        if strategy == 'tindex':
            return self.get_intersecting_tiles_from_index(geometry)
        elif strategy == 'first':
            first_file = self.get_first_raster_file()
            return [first_file] if first_file else []
        elif strategy == 'all':
            return self.get_all_raster_files()
        else:
            # Fallback to tile index method
            return self.get_intersecting_tiles_from_index(geometry)
    
    def get_all_raster_files(self):
        """Get ALL raster files from the mosaic."""
        if not self.config or 'tiles' not in self.config:
            return []
            
        files = []
        tiles = self.config['tiles']
        
        for quadkey, file_list in tiles.items():
            if file_list and isinstance(file_list, list):
                files.extend(file_list)
        
        return files
    
    def create_mosaic_for_polygon(self, geometry, temp_dir=None):
        """Create a mosaic of intersecting raster tiles using tile index."""
        
        # Use tile index to find intersecting files
        intersecting_files = self.get_intersecting_tiles_from_index(geometry)
        
        if not intersecting_files:
            print("❌ No intersecting raster files found from tile index")
            return None
        
        print(f"🔧 Creating mosaic from {len(intersecting_files)} files found via tile index...")
        
        # Test accessibility and open valid datasets
        valid_datasets = []
        
        for i, raster_file in enumerate(intersecting_files):
            print(f"🔧 Opening file {i+1}/{len(intersecting_files)}: {os.path.basename(raster_file)}")
            
            try:
                is_s3 = raster_file.startswith('s3://')
                
                if is_s3:
                    s3_vsi_path = f"/vsis3/{raster_file[5:]}"
                    src = rasterio.open(s3_vsi_path)
                    valid_datasets.append(src)
                    print(f"   ✅ S3 file opened: {src.width}x{src.height}")
                else:
                    if os.path.exists(raster_file):
                        src = rasterio.open(raster_file)
                        valid_datasets.append(src)
                        print(f"   ✅ Local file opened: {src.width}x{src.height}")
                    else:
                        print(f"   ❌ Local file not found")
                        
            except Exception as e:
                print(f"   ❌ Could not open {raster_file}: {e}")
                continue
        
        if not valid_datasets:
            print("❌ No valid datasets could be opened")
            return None
        
        print(f"✅ Successfully opened {len(valid_datasets)} datasets")
        
        try:
            temp_dir = temp_dir or tempfile.gettempdir()
            
            if len(valid_datasets) == 1:
                # Single file - create a copy
                print("🔧 Single intersecting file, creating copy...")
                
                mosaic_filename = f"single_raster_{hash(str(geometry.bounds)) % 100000}.tif"
                mosaic_path = os.path.join(temp_dir, mosaic_filename)
                
                src = valid_datasets[0]
                data = src.read()
                meta = src.meta.copy()
                meta.update({'driver': 'GTiff', 'compress': 'lzw'})
                
                with rasterio.open(mosaic_path, 'w', **meta) as dst:
                    dst.write(data)
                
                print(f"✅ Single raster saved to: {mosaic_path}")
                
            else:
                # Multiple files - create mosaic
                print(f"🔧 Merging {len(valid_datasets)} raster datasets...")
                
                mosaic_array, mosaic_transform = merge(valid_datasets)
                
                first_dataset = valid_datasets[0]
                mosaic_meta = first_dataset.meta.copy()
                mosaic_meta.update({
                    'driver': 'GTiff',
                    'height': mosaic_array.shape[1],
                    'width': mosaic_array.shape[2],
                    'transform': mosaic_transform,
                    'compress': 'lzw'
                })
                
                mosaic_filename = f"mosaic_{hash(str(geometry.bounds)) % 100000}.tif"
                mosaic_path = os.path.join(temp_dir, mosaic_filename)
                
                with rasterio.open(mosaic_path, 'w', **mosaic_meta) as dst:
                    dst.write(mosaic_array)
                
                print(f"✅ Mosaic created: {mosaic_path}")
            
            # Test the created mosaic (UPDATED)
            print("🧪 Testing created mosaic with zonal statistics...")
            test_stats = calculate_zonal_stats_with_custom(
                geometry,
                mosaic_path,
                ['count', 'mean'],  # Simple test stats
                -9999.0
            )
            
            print(f"📊 Mosaic test stats: {test_stats}")
            
            return mosaic_path
                
        except Exception as e:
            print(f"❌ Error creating mosaic: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return None
            
        finally:
            # Close all opened datasets
            for src in valid_datasets:
                try:
                    src.close()
                except:
                    pass
    
    def get_raster_info(self):
        """Get basic raster information."""
        first_file = self.get_first_raster_file()
        
        if not first_file:
            print("⚠️  No first file available, using default raster info")
            return {
                'transform': None,
                'width': 1000,
                'height': 1000,
                'dtype': 'float32',
                'nodata': -9999,
                'crs': self.crs
            }
        
        try:
            if self._is_s3_path(first_file):
                s3_vsi_path = f"/vsis3/{first_file[5:]}"
                with rasterio.open(s3_vsi_path) as src:
                    info = {
                        'transform': src.transform,
                        'width': src.width,
                        'height': src.height,
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata,
                        'crs': src.crs
                    }
                    print(f"📊 Raster info from VSI: {src.width}x{src.height}, dtype={src.dtypes[0]}, nodata={src.nodata}")
                    return info
            else:
                if os.path.exists(first_file):
                    with rasterio.open(first_file) as src:
                        info = {
                            'transform': src.transform,
                            'width': src.width,
                            'height': src.height,
                            'dtype': src.dtypes[0],
                            'nodata': src.nodata,
                            'crs': src.crs
                        }
                        print(f"📊 Raster info from local: {src.width}x{src.height}, dtype={src.dtypes[0]}, nodata={src.nodata}")
                        return info
                        
        except Exception as e:
            print(f"⚠️  Could not get detailed raster info: {e}")
        
        # Fallback with known CRS
        print("⚠️  Using fallback raster info")
        return {
            'transform': None,
            'width': 1000,
            'height': 1000,
            'dtype': 'float32',
            'nodata': -9999,
            'crs': self.crs
        }


def validate_inputs(args):
    """Validate input arguments and files."""
    # Check polygon file exists
    if not Path(args.polygons).exists():
        raise FileNotFoundError(f"Polygon file not found: {args.polygons}")
    
    # Skip validation for utility functions
    if args.generate_index_files or args.count_polygons:
        return True
    
    # For processing, require raster and statistics inputs
    if not (args.rasters or args.mosaic):
        raise ValueError("Either -r/--rasters or -m/--mosaic must be specified for processing")
    
    if not args.statistics:
        raise ValueError("-s/--statistics must be specified for processing")
    
    if not args.output:
        raise ValueError("-o/--output must be specified for processing")
    
    # Check raster inputs
    if args.rasters:
        raster_paths = [r.strip() for r in args.rasters.split(',')]
        for raster_path in raster_paths:
            # Skip validation for S3 paths and JSON files (will be validated at runtime)
            if not raster_path.startswith('s3://') and not raster_path.endswith('.json') and not Path(raster_path).exists():
                raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    if args.mosaic:
        # Skip validation for S3 mosaic JSON files (will be validated at runtime)
        if not args.mosaic.startswith('s3://') and not Path(args.mosaic).exists():
            raise FileNotFoundError(f"Mosaic file not found: {args.mosaic}")
    
    # Check S3 support if needed
    s3_needed = False
    if args.rasters:
        raster_paths = [r.strip() for r in args.rasters.split(',')]
        s3_needed = any(r.startswith('s3://') or r.endswith('.json') for r in raster_paths)
    if args.mosaic and (args.mosaic.startswith('s3://') or args.mosaic.endswith('.json')):
        s3_needed = True
    if args.tindex and args.tindex.startswith('s3://'):
        s3_needed = True
    
    if s3_needed and not S3_AVAILABLE:
        raise RuntimeError("boto3 is required for S3 support. Install with: pip install boto3")
    
    # Check index file if specified
    if args.index_file and not Path(args.index_file).exists():
        raise FileNotFoundError(f"Index file not found: {args.index_file}")
    
     # Validate statistics (updated to include 'nmad' and percentiles)
    if args.statistics:
        valid_builtin_stats = {'count', 'min', 'max', 'mean', 'sum', 'std', 'median', 'majority', 'minority'}
        requested_stats = [s.strip().lower() for s in args.statistics.split(',')]
        
        invalid_stats = []
        for stat in requested_stats:
            if stat not in valid_builtin_stats and stat != 'nmad':
                # Check if it's a valid percentile (p followed by digits)
                if not (stat.startswith('p') and stat[1:].isdigit() and 0 <= int(stat[1:]) <= 100):
                    invalid_stats.append(stat)
        
        if invalid_stats:
            raise ValueError(f"Invalid statistics: {invalid_stats}. Valid options: {valid_builtin_stats}, 'nmad', or percentiles like 'p10', 'p25', 'p50', 'p90'")
    
    
    # Validate processes
    if args.processes < 1:
        raise ValueError("Number of processes must be at least 1")
    
    return True

def parse_polygon_indices(args, total_polygons: int) -> Optional[List[int]]:
    """Parse polygon indices from various input methods."""
    if args.indices:
        # Parse comma-separated indices
        try:
            indices = [int(idx.strip()) for idx in args.indices.split(',')]
            # Validate indices
            invalid_indices = [idx for idx in indices if idx < 0 or idx >= total_polygons]
            if invalid_indices:
                raise ValueError(f"Invalid indices (must be 0-{total_polygons-1}): {invalid_indices}")
            return sorted(list(set(indices)))  # Remove duplicates and sort
        except ValueError as e:
            raise ValueError(f"Error parsing indices: {e}")
    
    elif args.index_range:
        # Parse range (e.g., "100-199")
        try:
            if '-' not in args.index_range:
                raise ValueError("Range must be in format 'start-end'")
            start_str, end_str = args.index_range.split('-', 1)
            start_idx = int(start_str.strip())
            end_idx = int(end_str.strip())
            
            if start_idx < 0 or end_idx >= total_polygons or start_idx > end_idx:
                raise ValueError(f"Invalid range: must be within 0-{total_polygons-1} and start <= end")
            
            return list(range(start_idx, end_idx + 1))
        except ValueError as e:
            raise ValueError(f"Error parsing index range: {e}")
    
    elif args.index_file:
        # Read indices from file
        try:
            with open(args.index_file, 'r') as f:
                indices = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        try:
                            idx = int(line)
                            if idx < 0 or idx >= total_polygons:
                                raise ValueError(f"Invalid index on line {line_num}: {idx} (must be 0-{total_polygons-1})")
                            indices.append(idx)
                        except ValueError as e:
                            raise ValueError(f"Error parsing line {line_num}: {e}")
                
                return sorted(list(set(indices)))  # Remove duplicates and sort
        except IOError as e:
            raise IOError(f"Error reading index file: {e}")
    
    # No indices specified - process all polygons
    return None


def generate_index_files(polygons_path: str, batch_size: int, output_dir: str, verbose: bool = False):
    """Generate index files for HPC batch processing."""
    # Load polygons to get count
    polygons_gdf = gpd.read_file(polygons_path)
    total_polygons = len(polygons_gdf)
    
    print(f"Total polygons: {total_polygons}")
    print(f"Batch size: {batch_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate batches
    num_batches = (total_polygons + batch_size - 1) // batch_size  # Ceiling division
    print(f"Generating {num_batches} batch files...")
    
    batch_info = []
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size - 1, total_polygons - 1)
        
        # Create index file
        index_file = Path(output_dir) / f"batch_{batch_num:04d}_indices.txt"
        
        with open(index_file, 'w') as f:
            f.write(f"# Batch {batch_num}: indices {start_idx} to {end_idx}\n")
            f.write(f"# Total polygons in batch: {end_idx - start_idx + 1}\n")
            for idx in range(start_idx, end_idx + 1):
                f.write(f"{idx}\n")
        
        batch_info.append({
            'batch_num': batch_num,
            'index_file': str(index_file),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'count': end_idx - start_idx + 1
        })
        
        if verbose:
            print(f"  Created {index_file}: indices {start_idx}-{end_idx} ({end_idx - start_idx + 1} polygons)")
    
    # Create batch summary file
    summary_file = Path(output_dir) / "batch_summary.json"
    summary_data = {
        'total_polygons': total_polygons,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'batches': batch_info
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Generated {num_batches} batch files in {output_dir}")
    print(f"Summary saved to {summary_file}")


def get_raster_crs_and_info(raster_sources: Union[List[str], str], s3_manager=None, tile_index_manager=None, mosaic_strategy='tindex') -> Tuple[CRS, Dict]:
    """Get CRS and basic info from raster source(s)."""
    print(f"🔧 Getting CRS and info from raster sources...")
    
    if isinstance(raster_sources, str):
        # Single raster or mosaic JSON
        if raster_sources.endswith('.json'):
            print(f"📋 Processing single mosaic JSON: {raster_sources}")
            
            # Handle single mosaic JSON
            mosaic_handler = MosaicHandler(raster_sources, s3_manager, tile_index_manager)
            
            # The mosaic handler has already determined the CRS
            crs = mosaic_handler.crs
            info = mosaic_handler.get_raster_info()
            
            print(f"✅ Mosaic CRS determined: {crs}")
            print(f"✅ Mosaic info: {info['width']}x{info['height']}, nodata={info['nodata']}")
            
            return crs, info
        
        else:
            print(f"📄 Processing single raster: {raster_sources}")
            # Single raster file
            if S3FileManager.is_s3_path(raster_sources):
                local_path = s3_manager.download_file(raster_sources)
                with rasterio.open(local_path) as src:
                    info = {
                        'transform': src.transform,
                        'width': src.width,
                        'height': src.height,
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata
                    }
                    print(f"✅ Single raster CRS: {src.crs}")
                    return src.crs, info
            else:
                with rasterio.open(raster_sources) as src:
                    info = {
                        'transform': src.transform,
                        'width': src.width,
                        'height': src.height,
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata
                    }
                    print(f"✅ Single raster CRS: {src.crs}")
                    return src.crs, info
    else:
        # List of rasters or mosaic JSONs
        print(f"📑 Processing multiple sources: {len(raster_sources)} files")
        
        # Check if these are mosaic JSON files or regular rasters
        first_source = raster_sources[0]
        
        if first_source.endswith('.json'):
            print(f"📋 Multiple mosaic JSON files detected")
            # Handle multiple mosaic JSON files - use first one for CRS
            mosaic_handler = MosaicHandler(first_source, s3_manager, tile_index_manager)
            crs = mosaic_handler.crs
            info = mosaic_handler.get_raster_info()
            
            #print(f"✅ Multi-mosaic CRS (from first): {crs}")
            print(f"✅ Multi-mosaic info: {info['width']}x{info['height']}, nodata={info['nodata']}")
            
            return crs, info
        else:
            print(f"📑 Multiple raster files detected")
            # Multiple regular raster files - use first one for CRS
            first_raster = raster_sources[0]
            if S3FileManager.is_s3_path(first_raster):
                local_path = s3_manager.download_file(first_raster)
                with rasterio.open(local_path) as src:
                    info = {
                        'transform': src.transform,
                        'width': src.width,
                        'height': src.height,
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata
                    }
                    print(f"✅ Multi-raster CRS (from first): {src.crs}")
                    return src.crs, info
            else:
                with rasterio.open(first_raster) as src:
                    info = {
                        'transform': src.transform,
                        'width': src.width,
                        'height': src.height,
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata
                    }
                    print(f"✅ Multi-raster CRS (from first): {src.crs}")
                    return src.crs, info

def reproject_polygons(polygons_gdf: gpd.GeoDataFrame, target_crs: CRS) -> gpd.GeoDataFrame:
    """Reproject polygons to target CRS if needed."""
    if polygons_gdf.crs != target_crs:
        print(f"🔄 Reprojecting polygons from {polygons_gdf.crs} to {target_crs}")
        return polygons_gdf.to_crs(target_crs)
    print(f"✅ Polygons already in correct CRS")
    return polygons_gdf

def process_polygon_chunk(args_tuple: Tuple) -> pd.DataFrame:
    """Process a chunk of polygons for zonal statistics extraction."""
    (polygon_chunk, raster_sources, statistics, default_prefix, nodata_value, 
     chunk_id, s3_profile, temp_dir, mosaic_strategy, tindex_path, custom_prefixes) = args_tuple
    
    print(f"🔧 Worker {chunk_id}: Starting chunk processing with {len(polygon_chunk)} polygons")
    
    results = []
    s3_manager = None
    tile_index_manager = None
    temp_files_to_cleanup = []
    
    # Initialize S3 manager if needed
    if S3_AVAILABLE and (
        (isinstance(raster_sources, str) and (raster_sources.startswith('s3://') or raster_sources.endswith('.json'))) or
        (isinstance(raster_sources, list) and any(r.startswith('s3://') or r.endswith('.json') for r in raster_sources)) or
        (tindex_path and tindex_path.startswith('s3://'))
    ):
        try:
            print(f"🔧 Worker {chunk_id}: Initializing S3 manager...")
            s3_manager = S3FileManager(profile_name=s3_profile, temp_dir=temp_dir)
            print(f"✅ Worker {chunk_id}: S3 manager initialized")
        except Exception as e:
            print(f"❌ Worker {chunk_id}: Could not initialize S3 manager: {e}")
    
    # Initialize tile index manager if needed
    if tindex_path and mosaic_strategy == 'tindex':
        try:
            print(f"🔧 Worker {chunk_id}: Loading tile index...")
            tile_index_manager = TileIndexManager(tindex_path, s3_manager)
            print(f"✅ Worker {chunk_id}: Tile index loaded")
        except Exception as e:
            print(f"❌ Worker {chunk_id}: Could not load tile index: {e}")
    
    try:
        print(f"🔧 Worker {chunk_id}: Processing {len(polygon_chunk)} polygons...")
        
        for idx, (orig_idx, polygon_row) in enumerate(polygon_chunk.iterrows()):
            print(f"🔧 Worker {chunk_id}: Processing polygon {idx+1}/{len(polygon_chunk)} (original index: {orig_idx})")
            
            try:
                geometry = polygon_row.geometry
                print(f"🔧 Worker {chunk_id}: Polygon geometry bounds: {geometry.bounds}")
                
                # Extract statistics from raster(s)
                if isinstance(raster_sources, str):
                    # Single source
                    if raster_sources.endswith('.json'):
                        print(f"🔧 Worker {chunk_id}: Processing single mosaic JSON...")
                        
                        # Handle single mosaic JSON with tile index
                        mosaic_handler = MosaicHandler(raster_sources, s3_manager, tile_index_manager)
                        print(f"🔧 Worker {chunk_id}: Mosaic handler created")
                        
                        # Create a mosaic of intersecting tiles
                        print(f"🔧 Worker {chunk_id}: Creating polygon-specific mosaic...")
                        mosaic_file = mosaic_handler.create_mosaic_for_polygon(geometry, temp_dir)
                        
                        if mosaic_file:
                            temp_files_to_cleanup.append(mosaic_file)
                            print(f"🔧 Worker {chunk_id}: Running zonal_stats on mosaic: {mosaic_file}")
                            
                            # Extract statistics from mosaic (UPDATED with custom stats)
                            stats = calculate_zonal_stats_with_custom(
                                geometry,
                                mosaic_file,
                                statistics,
                                nodata_value
                            )
                            
                            print(f"✅ Worker {chunk_id}: Single mosaic stats: {stats}")
                        else:
                            print(f"⚠️  Worker {chunk_id}: Could not create mosaic for polygon {orig_idx}")
                            stats = {stat: np.nan for stat in statistics}
                    
                    else:
                        # Single raster file processing
                        print(f"🔧 Worker {chunk_id}: Processing single raster file...")
                        is_s3 = raster_sources.startswith('s3://')
                        
                        if is_s3:
                            s3_vsi_path = f"/vsis3/{raster_sources[5:]}"
                            local_file = s3_vsi_path
                        else:
                            local_file = raster_sources
                        
                        # UPDATED to use custom function
                        stats = calculate_zonal_stats_with_custom(
                            geometry,
                            local_file,
                            statistics,
                            nodata_value
                        )
                        
                        print(f"✅ Worker {chunk_id}: Single file stats: {stats}")
                
                else:
                    # Multiple sources - could be rasters or mosaic JSONs
                    print(f"🔧 Worker {chunk_id}: Processing multiple sources ({len(raster_sources)} files)...")
                    
                    # Check if these are mosaic JSON files
                    if all(source.endswith('.json') for source in raster_sources):
                        print(f"🔧 Worker {chunk_id}: Multiple mosaic JSON files detected...")
                        
                        # Handle multiple mosaic JSON files
                        all_stats = {}
                        
                        for i, mosaic_source in enumerate(raster_sources):
                            print(f"🔧 Worker {chunk_id}: Processing mosaic {i+1}/{len(raster_sources)}: {os.path.basename(mosaic_source)}")
                            
                            try:
                                # Create mosaic handler for this JSON
                                mosaic_handler = MosaicHandler(mosaic_source, s3_manager, tile_index_manager)
                                
                                # Create mosaic for this polygon
                                mosaic_file = mosaic_handler.create_mosaic_for_polygon(geometry, temp_dir)
                                
                                if mosaic_file:
                                    temp_files_to_cleanup.append(mosaic_file)
                                    
                                    # Extract statistics from this mosaic (UPDATED with custom stats)
                                    mosaic_stats = calculate_zonal_stats_with_custom(
                                        geometry,
                                        mosaic_file,
                                        statistics,
                                        nodata_value
                                    )
                                    
                                    # Use custom prefix if provided, otherwise use default with index
                                    if custom_prefixes and i < len(custom_prefixes):
                                        prefix = custom_prefixes[i]
                                    else:
                                        prefix = f"{default_prefix}m{i+1}_"
                                    
                                    # Add stats with the appropriate prefix
                                    for stat_name, stat_value in mosaic_stats.items():
                                        all_stats[f"{prefix}{stat_name}"] = stat_value
                                    
                                    print(f"✅ Worker {chunk_id}: Mosaic {i+1} stats with prefix '{prefix}': {mosaic_stats}")
                                else:
                                    print(f"⚠️  Worker {chunk_id}: Could not create mosaic {i+1}")
                                    # Add NaN values for this mosaic
                                    if custom_prefixes and i < len(custom_prefixes):
                                        prefix = custom_prefixes[i]
                                    else:
                                        prefix = f"{default_prefix}m{i+1}_"
                                    
                                    for stat in statistics:
                                        all_stats[f"{prefix}{stat}"] = np.nan
                                            
                            except Exception as e:
                                print(f"❌ Worker {chunk_id}: Error processing mosaic {i+1}: {e}")
                                # Add NaN values for this mosaic
                                if custom_prefixes and i < len(custom_prefixes):
                                    prefix = custom_prefixes[i]
                                else:
                                    prefix = f"{default_prefix}m{i+1}_"
                                
                                for stat in statistics:
                                    all_stats[f"{prefix}{stat}"] = np.nan
                        
                        stats = all_stats
                        print(f"✅ Worker {chunk_id}: Multi-mosaic combined stats: {list(stats.keys())}")
                    
                    else:
                        print(f"🔧 Worker {chunk_id}: Multiple raster files detected...")
                        
                        # Handle multiple regular raster files
                        all_stats = {}
                        for i, raster_path in enumerate(raster_sources):
                            is_s3 = raster_path.startswith('s3://')
                            
                            if is_s3:
                                s3_vsi_path = f"/vsis3/{raster_path[5:]}"
                                local_file = s3_vsi_path
                            else:
                                local_file = raster_path
                            
                            # UPDATED to use custom function
                            raster_stats = calculate_zonal_stats_with_custom(
                                geometry,
                                local_file,
                                statistics,
                                nodata_value
                            )
                            
                            # Use custom prefix if provided, otherwise use default with index
                            if custom_prefixes and i < len(custom_prefixes):
                                prefix = custom_prefixes[i]
                            else:
                                prefix = f"{default_prefix}r{i+1}_"
                            
                            # Add stats with the appropriate prefix
                            for stat_name, stat_value in raster_stats.items():
                                all_stats[f"{prefix}{stat_name}"] = stat_value
                        
                        stats = all_stats
                        print(f"✅ Worker {chunk_id}: Multi-raster stats: {list(stats.keys())}")
                
                # Create result row
                result_row = polygon_row.to_dict()
                
                # Add statistics (they already have prefixes applied above for multiple sources)
                if isinstance(raster_sources, str):
                    # Single source - apply default prefix or custom prefix
                    if custom_prefixes and len(custom_prefixes) > 0:
                        prefix = custom_prefixes[0]
                    else:
                        prefix = default_prefix
                    
                    for stat_name, stat_value in stats.items():
                        if stat_value is not None and not (isinstance(stat_value, float) and np.isnan(stat_value)):
                            result_row[f"{prefix}{stat_name}"] = stat_value
                        else:
                            result_row[f"{prefix}{stat_name}"] = np.nan
                else:
                    # Multiple sources - prefixes already applied in stats dict
                    for stat_name, stat_value in stats.items():
                        if stat_value is not None and not (isinstance(stat_value, float) and np.isnan(stat_value)):
                            result_row[stat_name] = stat_value
                        else:
                            result_row[stat_name] = np.nan
                
                result_row['_original_index'] = orig_idx
                results.append(result_row)
                
                print(f"✅ Worker {chunk_id}: Polygon {orig_idx} completed successfully")
                
            except Exception as e:
                print(f"❌ Worker {chunk_id}: Error processing polygon {orig_idx}: {str(e)}")
                import traceback
                print(f"❌ Worker {chunk_id}: Traceback: {traceback.format_exc()}")
                
                # Add row with NaN statistics
                result_row = polygon_row.to_dict()
                
                # Handle NaN statistics with appropriate prefixes
                if isinstance(raster_sources, str):
                    # Single source
                    if custom_prefixes and len(custom_prefixes) > 0:
                        prefix = custom_prefixes[0]
                    else:
                        prefix = default_prefix
                    
                    for stat in statistics:
                        result_row[f"{prefix}{stat}"] = np.nan
                else:
                    # Multiple sources
                    for i in range(len(raster_sources)):
                        if custom_prefixes and i < len(custom_prefixes):
                            prefix = custom_prefixes[i]
                        else:
                            if raster_sources[i].endswith('.json'):
                                prefix = f"{default_prefix}m{i+1}_"
                            else:
                                prefix = f"{default_prefix}r{i+1}_"
                        
                        for stat in statistics:
                            result_row[f"{prefix}{stat}"] = np.nan
                
                result_row['_original_index'] = orig_idx
                results.append(result_row)
        
        print(f"✅ Worker {chunk_id}: Chunk processing completed. {len(results)} results.")
    
    except Exception as e:
        print(f"❌ Worker {chunk_id}: Fatal error in chunk processing: {e}")
        import traceback
        print(f"❌ Worker {chunk_id}: Traceback: {traceback.format_exc()}")
    
    finally:
        # Clean up temporary mosaic files
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🔧 Worker {chunk_id}: Cleaned up temp file: {temp_file}")
            except Exception as e:
                print(f"⚠️  Worker {chunk_id}: Could not clean up {temp_file}: {e}")
        
        # Clean up S3 temporary files
        if s3_manager:
            print(f"🔧 Worker {chunk_id}: Cleaning up S3 manager...")
            s3_manager.cleanup()
    
    print(f"🏁 Worker {chunk_id}: Returning {len(results)} results")
    return pd.DataFrame(results)

def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate inputs
        validate_inputs(args)
        
        if args.verbose:
            print("Loading polygon data...")
        
        # Load polygons
        polygons_gdf = gpd.read_file(args.polygons)
        total_polygons = len(polygons_gdf)
        
        # Handle utility functions
        if args.count_polygons:
            print(f"Total polygon count: {total_polygons}")
            return
        
        if args.generate_index_files:
            generate_index_files(args.polygons, args.batch_size, args.output_dir, args.verbose)
            return
        
        print(f"Loaded {total_polygons} polygons from {args.polygons}")
        
        # Parse polygon indices if specified
        polygon_indices = parse_polygon_indices(args, total_polygons)
        
        if polygon_indices is not None:
            print(f"Processing {len(polygon_indices)} selected polygons (indices: {min(polygon_indices)}-{max(polygon_indices)})")
            # Subset polygons to selected indices
            polygons_gdf = polygons_gdf.iloc[polygon_indices].copy()
            # Store original indices if requested
            if args.preserve_index:
                polygons_gdf['original_index'] = polygon_indices
        else:
            print("Processing all polygons")
            if args.preserve_index:
                polygons_gdf['original_index'] = range(len(polygons_gdf))
        
        # Initialize S3 manager if needed (updated condition)
        s3_manager = None
        s3_needed = False
        
        # Check if S3 is needed
        if args.rasters:
            raster_paths = [r.strip() for r in args.rasters.split(',')]
            s3_needed = any(r.startswith('s3://') for r in raster_paths)
        if args.mosaic and args.mosaic.startswith('s3://'):
            s3_needed = True
        if args.tindex and args.tindex.startswith('s3://'):
            s3_needed = True
        
        if S3_AVAILABLE and s3_needed:
            try:
                s3_manager = S3FileManager(profile_name=args.s3_profile, temp_dir=args.temp_dir)
                if args.verbose:
                    print("✅ S3 manager initialized successfully")
            except Exception as e:
                print(f"❌ Could not initialize S3 manager: {e}")
                if s3_needed:
                    print("❌ S3 manager is required for S3 paths")
                    return
        elif s3_needed:
            print("❌ boto3 is required for S3 support but not available")
            return
        
        # Initialize tile index manager if needed
        tile_index_manager = None
        if args.tindex and args.mosaic_strategy == 'tindex':
            try:
                print(f"🔧 Loading tile index: {args.tindex}")
                tile_index_manager = TileIndexManager(args.tindex, s3_manager)
            except Exception as e:
                print(f"⚠️  Could not load tile index: {e}")
                if args.mosaic_strategy == 'tindex':
                    print("❌ Tile index strategy requires valid tile index")
                    return
        
        # Parse raster sources
        if args.rasters:
            raster_sources = [r.strip() for r in args.rasters.split(',')]
            if len(raster_sources) == 1:
                raster_sources = raster_sources[0]
        else:
            raster_sources = args.mosaic
        
        # Validate custom prefixes AFTER we know the raster sources
        custom_prefixes = None
        if args.zs_col_prefix:
            custom_prefixes = [p.strip() for p in args.zs_col_prefix.split(',')]
            
            # Count raster sources
            if isinstance(raster_sources, str):
                num_sources = 1
            else:
                num_sources = len(raster_sources)
            
            if len(custom_prefixes) != num_sources:
                raise ValueError(f"Number of custom prefixes ({len(custom_prefixes)}) must match number of raster sources ({num_sources})")
            
            # Validate prefix format (should end with underscore for consistency)
            for i, prefix in enumerate(custom_prefixes):
                if not prefix.endswith('_'):
                    custom_prefixes[i] = prefix + '_'
            
            print(f"Using custom prefixes: {custom_prefixes}")
        
        # Get raster CRS and info
        print(f"\n🔧 Determining raster CRS and info...")
        raster_crs, raster_info = get_raster_crs_and_info(raster_sources, s3_manager, tile_index_manager, args.mosaic_strategy)
        
        #print(f"\n📐 FINAL RASTER CRS: {raster_crs}")
        print(f"📊 RASTER INFO: {raster_info['width']}x{raster_info['height']}, dtype={raster_info['dtype']}, nodata={raster_info['nodata']}")
        
        # Reproject polygons if necessary
        original_crs = polygons_gdf.crs
        print(f"📍 Polygon CRS: {original_crs}")
        
        if original_crs != raster_crs:
            polygons_gdf = reproject_polygons(polygons_gdf, raster_crs)
            print(f"✅ Polygons reprojected successfully")
        else:
            print(f"✅ Polygons already in correct CRS")
        
        # Parse statistics and columns
        statistics = [s.strip().lower() for s in args.statistics.split(',')]
        columns_to_keep = [c.strip() for c in args.columns.split(',')] if args.columns else []
        
        # Set up nodata value
        nodata_value = args.nodata if args.nodata is not None else raster_info.get('nodata')
        
        if args.verbose:
            print(f"Statistics to compute: {statistics}")
            print(f"Using {args.processes} processes")
            print(f"Chunk size: {args.chunk_size}")
            print(f"Mosaic strategy: {args.mosaic_strategy}")
            if nodata_value is not None:
                print(f"NoData value: {nodata_value}")
        
        # Split polygons into chunks for parallel processing
        polygon_chunks = []
        for i in range(0, len(polygons_gdf), args.chunk_size):
            chunk = polygons_gdf.iloc[i:i+args.chunk_size]
            chunk_args = (
                chunk, raster_sources, statistics, args.prefix, nodata_value,
                i//args.chunk_size, args.s3_profile, args.temp_dir, args.mosaic_strategy, 
                args.tindex, custom_prefixes  # Add custom_prefixes to chunk args
            )
            polygon_chunks.append(chunk_args)
        
        print(f"Processing {len(polygon_chunks)} chunks...")
        
        # Process chunks in parallel
        if args.processes == 1:
            # Single process for debugging
            results = []
            for chunk_args in tqdm(polygon_chunks, desc="Processing chunks"):
                result = process_polygon_chunk(chunk_args)
                results.append(result)
        else:
            # Multiprocessing
            with mp.Pool(processes=args.processes) as pool:
                results = list(tqdm(
                    pool.imap(process_polygon_chunk, polygon_chunks),
                    total=len(polygon_chunks),
                    desc="Processing chunks"
                ))
        
        # Combine results
        if args.verbose:
            print("Combining results...")
        
        combined_results = pd.concat(results, ignore_index=True)
        
        # Sort by original index to maintain order
        combined_results = combined_results.sort_values('_original_index')
        combined_results = combined_results.drop('_original_index', axis=1)
        
        # Create output GeoDataFrame
        output_gdf = gpd.GeoDataFrame(combined_results, crs=raster_crs)
        
        # Filter columns if specified
        if columns_to_keep:
            # Keep specified columns plus geometry and statistics columns
            # Determine what prefixes were used for filtering
            if custom_prefixes:
                all_prefixes = custom_prefixes
            else:
                if isinstance(raster_sources, str):
                    all_prefixes = [args.prefix]
                else:
                    if all(source.endswith('.json') for source in raster_sources):
                        all_prefixes = [f"{args.prefix}m{i+1}_" for i in range(len(raster_sources))]
                    else:
                        all_prefixes = [f"{args.prefix}r{i+1}_" for i in range(len(raster_sources))]
            
            stat_columns = []
            for prefix in all_prefixes:
                stat_columns.extend([col for col in output_gdf.columns if col.startswith(prefix)])
            
            columns_to_keep_final = ['geometry'] + columns_to_keep + stat_columns
            if args.preserve_index and 'original_index' in output_gdf.columns:
                columns_to_keep_final.append('original_index')
            columns_to_keep_final = [col for col in columns_to_keep_final if col in output_gdf.columns]
            output_gdf = output_gdf[columns_to_keep_final]
        
        # Save output
        if args.verbose:
            print(f"Saving results to {args.output}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save to geopackage
        output_gdf.to_file(args.output, driver='GPKG')
        
        print(f"Successfully processed {len(output_gdf)} polygons")
        print(f"Output saved to: {args.output}")
        
        # Print summary statistics
        if args.verbose:
            # Determine what prefixes were used for summary
            if custom_prefixes:
                all_prefixes = custom_prefixes
            else:
                if isinstance(raster_sources, str):
                    all_prefixes = [args.prefix]
                else:
                    if all(source.endswith('.json') for source in raster_sources):
                        all_prefixes = [f"{args.prefix}m{i+1}_" for i in range(len(raster_sources))]
                    else:
                        all_prefixes = [f"{args.prefix}r{i+1}_" for i in range(len(raster_sources))]
            
            stat_columns = []
            for prefix in all_prefixes:
                stat_columns.extend([col for col in output_gdf.columns if col.startswith(prefix)])
            
            print(f"\nStatistics columns added: {len(stat_columns)}")
            for col in stat_columns:
                non_null_count = output_gdf[col].notna().sum()
                print(f"  {col}: {non_null_count}/{len(output_gdf)} non-null values")
        
        # Clean up S3 manager if used
        if s3_manager:
            s3_manager.cleanup()
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
def run_zonal_stats(args_list):
    """
    Run zonal statistics with a list of arguments (for programmatic use).
    
    Parameters:
    -----------
    args_list : list
        List of command line arguments (without script name)
        
    Example:
    --------
    run_zonal_stats(['-p', 'polygons.gpkg', '-m', 'mosaic.json', '-s', 'mean,std', '-o', 'output.gpkg'])
    """
    import sys
    # Temporarily replace sys.argv
    original_argv = sys.argv
    try:
        sys.argv = ['zonal_stats_landscapes.py'] + args_list
        main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()