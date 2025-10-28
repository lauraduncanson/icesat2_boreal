#!/usr/bin/env python3

import numpy as np
import rasterio
from rasterio.plot import show
import glob
from scipy.stats import theilslopes, kendalltau, pearsonr, linregress, t
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
from CovariateUtils import write_cog

# # Bayesian imports
# import pymc as pm
# import arviz as az
# import pytensor.tensor as pt

# def read_date_info(date_raster_file):
#     """
#     Read JulianDate and yearDate from a date raster file
#     Returns combined date in decimal years
#     """
#     with rasterio.open(date_raster_file) as src:
#         julian_date = src.read(1)  # JulianDate layer
#         year_date = src.read(2)    # yearDate layer
    
#     # Convert to decimal years (year + julian_day/365.25)
#     decimal_date = year_date + (julian_date / 365.25)
#     return decimal_date

def process_chunk_breakpoints(args):
    """
    Process a chunk of pixels for breakpoint detection using multiprocessing
    """
    from ruptures import Pelt
    import numpy as np
    
    chunk_values, chunk_dates, start_row, start_col, chunk_rows, chunk_cols, min_break_position, penalty, nodata_value = args
    
    results = []
    for i in range(chunk_rows):
        for j in range(chunk_cols):
            pixel_values = chunk_values[:, i, j]
            pixel_dates = chunk_dates[:, i, j]
            pixel_idx = ((start_row + i), (start_col + j))
            
            result = compute_pixel_breakpoint_statistics(
                (pixel_values, pixel_dates, pixel_idx, min_break_position, penalty, nodata_value)
            )
            results.append(result)
    
    return results

def compute_pixel_breakpoint_statistics(args):
    """
    Compute breakpoint statistics for a single pixel
    """
    from ruptures import Pelt
    import numpy as np
    
    pixel_values, pixel_dates, pixel_idx, min_break_position, penalty, nodata_value = args
    
    # Initialize result dictionary
    result = {
        'break_detected': 0,
        'break_year_index': -1,
        'break_year': nodata_value,
        'break_magnitude': np.nan
    }
    
    try:
        # Skip if not enough valid data
        valid_mask = ~np.isnan(pixel_values)
        if np.sum(valid_mask) < 5:
            return result
        
        ts = pixel_values.copy()
        dates = pixel_dates.copy()
        
        # Handle missing values with interpolation if needed
        if np.sum(valid_mask) < len(ts):
            valid_indices = np.where(valid_mask)[0]
            all_indices = np.arange(len(ts))
            if len(valid_indices) >= 2:
                ts = np.interp(all_indices, valid_indices, ts[valid_indices])
                # Use corresponding dates for interpolated values
                dates = np.interp(all_indices, valid_indices, dates[valid_indices])
        
        # Run PELT breakpoint detection
        model = Pelt(model="l2").fit(ts.reshape(-1, 1))
        breakpoints = model.predict(pen=penalty)
        
        # If breakpoints were found (other than the end of series)
        if len(breakpoints) > 1:
            # Get the last breakpoint (most recent abrupt change)
            last_break = breakpoints[-2]  # -2 because last element is series length
            
            # Only record if in the latter part of the series
            if last_break > min_break_position * len(ts):
                result['break_detected'] = 1
                result['break_year_index'] = last_break
                result['break_year'] = dates[last_break]
                
                # Calculate magnitude of change (drop in biomass)
                if last_break < len(ts) - 1:
                    before_avg = np.mean(ts[max(0, last_break-2):last_break])
                    after_avg = np.mean(ts[last_break:min(len(ts), last_break+2)])
                    result['break_magnitude'] = after_avg - before_avg
    
    except Exception as e:
        # Return default values on error
        pass
    
    return result

def detect_breakpoints_raster(value_raster_files, DATE_RASTER_FILES, OUT_FILE, 
                             chunk_size=300, min_break_position=0.7, penalty=10, n_processes=None, nodata_value=-9999):
    """
    Detect breakpoints in raster time series data and write results to COG using multiprocessing
    
    Parameters:
    -----------
    value_raster_files : list
        List of paths to value raster files
    DATE_RASTER_FILES : list
        List of paths to corresponding date raster files
    OUT_FILE : str
        Base output file name
    TREND_METHOD : str
        Trend method name for reference file
    chunk_size : int
        Processing chunk size for memory efficiency
    min_break_position : float
        Minimum position in time series to consider breaks (0.7 = last 30%)
    penalty : int
        PELT penalty parameter (higher = fewer breakpoints)
    n_processes : int
        Number of processes for parallel computation
    
    Returns:
    --------
    str : Path to output breakpoint COG file
    """
    from multiprocessing import Pool, cpu_count
    import numpy as np
    import rasterio
    
    if n_processes is None:
        n_processes = min(30, cpu_count())
    
    print(f"Running breakpoint detection analysis using {n_processes} processes...")
    
    # Get metadata from reference file for the COG
    with rasterio.open(value_raster_files[0]) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
    
    print(f"Raster dimensions: {height} x {width}")
    
    # Read all rasters to get the time series for each pixel
    print("Reading rasters for breakpoint detection...")
    n_rasters = len(value_raster_files)
    value_stack = np.zeros((n_rasters, height, width), dtype=np.float32)
    date_stack = np.zeros((n_rasters, height, width), dtype=np.float32)
    
    for i, (value_file, date_file) in enumerate(zip(value_raster_files, DATE_RASTER_FILES)):
        print(f"  Reading pair {i+1}/{n_rasters}")
        
        # Read value raster
        with rasterio.open(value_file) as src:
            value_stack[i] = src.read(1)
        
        # Read date information with nodata handling
        date_data = read_date_info(date_file)
        # Check if date_data has nodata values and replace with NaN
        date_data = np.where(date_data < 1900, np.nan, date_data)  # Years should be > 2000
        date_data = np.where(date_data > 2099, np.nan, date_data)  # Years should be < 2030
        date_data = np.where(date_data == nodata_value, np.nan, date_data)  # Handle explicit nodata
        date_stack[i] = date_data
    
    print("Preparing data for parallel breakpoint processing...")
    
    # Initialize output arrays
    output_arrays = {
        'break_detected': np.zeros((height, width), dtype=np.uint8),
        'break_year_index': np.full((height, width), -1, dtype=np.int16),
        'break_year': np.full((height, width), nodata_value, dtype=np.float32),
        'break_magnitude': np.full((height, width), np.nan, dtype=np.float32)
    }
    
    # Process in chunks for memory efficiency
    chunk_args = []
    total_chunks = 0
    
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk_rows = end_row - start_row
        
        for start_col in range(0, width, chunk_size):
            end_col = min(start_col + chunk_size, width)
            chunk_cols = end_col - start_col
            
            # Extract chunk data
            chunk_values = value_stack[:, start_row:end_row, start_col:end_col]
            chunk_dates = date_stack[:, start_row:end_row, start_col:end_col]
            
            chunk_args.append((chunk_values, chunk_dates, start_row, start_col, 
                             chunk_rows, chunk_cols, min_break_position, penalty, nodata_value))
            total_chunks += 1
    
    print(f"Processing {total_chunks} chunks in parallel for breakpoint detection...")
    
    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        chunk_results = []
        for i, result in enumerate(pool.imap(process_chunk_breakpoints, chunk_args)):
            chunk_results.append(result)
            if (i + 1) % max(1, total_chunks // 10) == 0:
                print(f"  Completed {i+1}/{total_chunks} chunks ({100*(i+1)/total_chunks:.1f}%)")
    
    # Reassemble results
    print("Assembling breakpoint results...")
    chunk_idx = 0
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        
        for start_col in range(0, width, chunk_size):
            end_col = min(start_col + chunk_size, width)
            
            chunk_result = chunk_results[chunk_idx]
            result_idx = 0
            
            for i in range(start_row, end_row):
                for j in range(start_col, end_col):
                    result = chunk_result[result_idx]
                    
                    # Fill output arrays
                    for key in output_arrays.keys():
                        if key in result:
                            output_arrays[key][i, j] = result[key]
                    
                    result_idx += 1
            
            chunk_idx += 1
    
    # Create a COG with the breakpoint results
    print("Creating breakpoint COG...")
    
    # Stack the results into bands
    break_stack = np.stack([
        output_arrays['break_detected'],
        output_arrays['break_year_index'].astype(np.float32),  # Convert to float32 for consistency
        output_arrays['break_year'],
        output_arrays['break_magnitude']
    ])
    
    stack_names = [
        'break_detected',
        'break_year_index', 
        'break_year',
        'break_magnitude'
    ]
    
    # Write the COG
    break_output_file = f"{OUT_FILE}_breakpoints.tif"
    write_cog(
        break_stack,
        break_output_file,
        crs,
        transform,
        stack_names,
        out_crs=crs,
        input_nodata_value=nodata_value,
        resampling='nearest'
    )
    
    # Print summary statistics
    print("\n" + "="*50)
    print("BREAKPOINT DETECTION SUMMARY")
    print("="*50)
    
    total_pixels = height * width
    break_pixels = np.sum(output_arrays['break_detected'])
    break_percentage = (break_pixels / total_pixels) * 100
    
    print(f"Total pixels: {total_pixels}")
    print(f"Pixels with breakpoints detected: {break_pixels} ({break_percentage:.2f}%)")
    
    if break_pixels > 0:
        valid_years = output_arrays['break_year'][output_arrays['break_year'] != nodata_value]
        valid_magnitudes = output_arrays['break_magnitude'][~np.isnan(output_arrays['break_magnitude'])]
        
        if len(valid_years) > 0:
            print(f"Break year range: {np.min(valid_years):.1f} to {np.max(valid_years):.1f}")
        
        if len(valid_magnitudes) > 0:
            print(f"Break magnitude stats:")
            print(f"  Mean: {np.mean(valid_magnitudes):.6f}")
            print(f"  Std:  {np.std(valid_magnitudes):.6f}")
            print(f"  Min:  {np.min(valid_magnitudes):.6f}")
            print(f"  Max:  {np.max(valid_magnitudes):.6f}")
    
    print(f"Breakpoint detection complete. Results saved to: {break_output_file}")
    
    return break_output_file
    
def read_date_info(date_raster_file, year_layer='yearDate', julian_layer='JulianDate'):
    """
    Read JulianDate and yearDate from a date raster file using layer names
    
    Parameters:
    date_raster_file: path to the raster file containing date information
    year_layer: name of the layer containing year data
    julian_layer: name of the layer containing Julian day data
    
    Returns:
    Combined date in decimal years
    """
    with rasterio.open(date_raster_file) as src:
        # Get band indices from layer names
        band_names = src.descriptions or [f"Band {i+1}" for i in range(src.count)]
        
        # Find the indices for the specified layers
        year_idx = None
        julian_idx = None
        
        # Try to find exact matches first
        for i, name in enumerate(band_names):
            if name == year_layer:
                year_idx = i + 1  # rasterio bands are 1-indexed
            elif name == julian_layer:
                julian_idx = i + 1
        
        # If not found, try case-insensitive substring matching
        if year_idx is None or julian_idx is None:
            for i, name in enumerate(band_names):
                if year_idx is None and year_layer.lower() in name.lower():
                    year_idx = i + 1
                if julian_idx is None and julian_layer.lower() in name.lower():
                    julian_idx = i + 1
        
        # If still not found, check metadata tags for layer information
        if year_idx is None or julian_idx is None:
            for i in range(src.count):
                tags = src.tags(i+1) or {}
                if year_idx is None and any(year_layer.lower() in str(v).lower() for v in tags.values()):
                    year_idx = i + 1
                if julian_idx is None and any(julian_layer.lower() in str(v).lower() for v in tags.values()):
                    julian_idx = i + 1
        
        # Default to first two bands if no matches found
        if year_idx is None:
            print(f"Warning: Couldn't find layer '{year_layer}', using band 2")
            year_idx = 2
        if julian_idx is None:
            print(f"Warning: Couldn't find layer '{julian_layer}', using band 1")
            julian_idx = 1
        
        # Read the bands
        julian_date = src.read(julian_idx)
        year_date = src.read(year_idx)
        
        print(f"Using band {julian_idx} for Julian days and band {year_idx} for years in {date_raster_file}")
    
    # Convert to decimal years (year + julian_day/365.25)
    decimal_date = year_date + (julian_date / 365.25)
    return decimal_date

def ols_slopes(y_values, x_values, alpha=0.05):
    """
    OLS equivalent to theilslopes
    Returns: slope, intercept, slope_lower_ci, slope_upper_ci
    """
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
    
    n = len(y_values)
    df = n - 2
    t_val = t.ppf(1 - alpha/2, df)
    
    slope_lo = slope - t_val * std_err
    slope_hi = slope + t_val * std_err
    
    return slope, intercept, slope_lo, slope_hi

def compute_pixel_statistics(args):
    """
    Compute comprehensive trend statistics for a single pixel
    """
    pixel_values, pixel_dates, pixel_idx, do_ols, nodata_value = args
    
    # Initialize results dictionary
    results = {
        'trendslope': np.nan,
        'trendintercept': np.nan,
        'r2': np.nan,
        'kendall_tau': np.nan,
        'kendall_pvalue': np.nan,
        'n_obs': 0,
        'trendslope_lo': np.nan,
        'trendslope_hi': np.nan,
        'pixel_idx': pixel_idx
    }
    
    # Remove invalid data points
    valid_mask = ~(np.isnan(pixel_values) | np.isnan(pixel_dates) | 
                   (pixel_values == nodata_value) | (pixel_dates == nodata_value))
    
    if np.sum(valid_mask) < 3:  # Need at least 3 points for trend
        return results
    
    valid_values = pixel_values[valid_mask]
    valid_dates = pixel_dates[valid_mask]
    results['n_obs'] = len(valid_values)
    
    try:
        if do_ols:
            # OLS trend estimation:
            trendslope, trendintercept, trendslope_lo, trendslope_hi = ols_slopes(valid_values, valid_dates)
        else:
            # Theil-Sen trend estimation
            trendslope, trendintercept, trendslope_lo, trendslope_hi = theilslopes(valid_values, valid_dates)
            
        results['trendslope'] = trendslope
        results['trendintercept'] = trendintercept
        results['trendslope_lo'] = trendslope_lo
        results['trendslope_hi'] = trendslope_hi
        
        # Kendall's tau test (non-parametric correlation).
        tau, kendall_p = kendalltau(valid_dates, valid_values)
        # rank correlation coefficient that measures the strength and direction of monotonic
        # relationships between two variables
        results['kendall_tau'] = tau
        # p-value for significance of Theil-Sen slope (using Kendall's tau)
        results['kendall_pvalue'] = kendall_p
        
        # R-squared calculation (correlation between observed and predicted)
        predicted_values = trendslope * valid_dates + trendintercept
        if len(valid_values) > 1:
            correlation, _ = pearsonr(valid_values, predicted_values)
            results['r2'] = correlation ** 2 if not np.isnan(correlation) else np.nan
            
    except Exception as e:
        # Keep NaN values if computation fails
        pass
    
    return results

def process_chunk(args):
    """
    Process a chunk of pixels using multiprocessing
    """
    chunk_data, chunk_dates, start_row, start_col, chunk_rows, chunk_cols, do_ols, nodata_value = args
    
    results = []
    for i in range(chunk_rows):
        for j in range(chunk_cols):
            pixel_values = chunk_data[:, i, j]
            pixel_dates = chunk_dates[:, i, j]
            pixel_idx = ((start_row + i), (start_col + j))
            
            result = compute_pixel_statistics((pixel_values, pixel_dates, pixel_idx, do_ols, nodata_value))
            results.append(result)
    
    return results

def stack_rasters_and_compute_trend(value_raster_files, date_raster_files, 
                                                   band_num=1, output_file=None, n_processes=None, 
                                                   chunk_size=500, do_ols=False, nodata_value=-9999):
    """
    Stack multiple rasters with corresponding date info and compute trends: OLS or Theil-Sen statistics
    
    Parameters:
    value_raster_files: list of paths to value rasters (band 1 will be used)
    date_raster_files: list of paths to corresponding date rasters (JulianDate, yearDate)
    output_file: base name for output files
    n_processes: number of processes for parallel computation
    chunk_size: size of processing chunks
    """
    
    if n_processes is None:
        n_processes = min(30, cpu_count())  # Use up to 30 CPUs as specified
    
    print(f"Using {n_processes} processes for computation")
    
    # Validate input
    if len(value_raster_files) != len(date_raster_files):
        raise ValueError("Number of value and date rasters must match")
    
    n_rasters = len(value_raster_files)
    print(f"Processing {n_rasters} raster pairs...")
    
    # Read the first raster to get metadata
    with rasterio.open(value_raster_files[0]) as src:
        profile = src.profile
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
    
    print(f"Raster dimensions: {height} x {width}")
    
    # Initialize arrays for all data
    value_stack = np.zeros((n_rasters, height, width), dtype=np.float32)
    date_stack = np.zeros((n_rasters, height, width), dtype=np.float32)
    
    # Read all rasters
    print("Reading value and date rasters...")
    for i, (value_file, date_file) in enumerate(zip(value_raster_files, date_raster_files)):
        print(f"  Reading pair {i+1}/{n_rasters}")
        
        # Read value raster from band
        with rasterio.open(value_file) as src:
            value_stack[i] = src.read(band_num)
        
        # Read date information
        date_stack[i] = read_date_info(date_file)
    
    print("Preparing data for parallel processing...")
    
    # Initialize output arrays
    output_arrays = {
        'trendslope':     np.full((height, width), np.nan, dtype=np.float32),
        'trendintercept': np.full((height, width), np.nan, dtype=np.float32),
        'r2':             np.full((height, width), np.nan, dtype=np.float32),
        'kendall_tau':    np.full((height, width), np.nan, dtype=np.float32),
        'kendall_pvalue': np.full((height, width), np.nan, dtype=np.float32),
        # 'n_obs':          np.full((height, width), 0, dtype=np.int16),
        'n_obs':          np.full((height, width), 0, dtype=np.float32),
        'trendslope_lo':  np.full((height, width), np.nan, dtype=np.float32),
        'trendslope_hi':  np.full((height, width), np.nan, dtype=np.float32)
    }
    
    # Process in chunks for memory efficiency
    chunk_args = []
    total_chunks = 0
    
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk_rows = end_row - start_row
        
        for start_col in range(0, width, chunk_size):
            end_col = min(start_col + chunk_size, width)
            chunk_cols = end_col - start_col
            
            # Extract chunk data
            chunk_values = value_stack[:, start_row:end_row, start_col:end_col]
            chunk_dates = date_stack[:, start_row:end_row, start_col:end_col]
            
            chunk_args.append((chunk_values, chunk_dates, start_row, start_col, 
                             chunk_rows, chunk_cols, do_ols, nodata_value))
            total_chunks += 1
    
    print(f"Processing {total_chunks} chunks in parallel...")
    
    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        chunk_results = []
        for i, result in enumerate(pool.imap(process_chunk, chunk_args)):
            chunk_results.append(result)
            if (i + 1) % max(1, total_chunks // 10) == 0:
                print(f"  Completed {i+1}/{total_chunks} chunks ({100*(i+1)/total_chunks:.1f}%)")
    
    # Reassemble results
    print("Assembling results...")
    chunk_idx = 0
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        
        for start_col in range(0, width, chunk_size):
            end_col = min(start_col + chunk_size, width)
            
            chunk_result = chunk_results[chunk_idx]
            result_idx = 0
            
            for i in range(start_row, end_row):
                for j in range(start_col, end_col):
                    result = chunk_result[result_idx]
                    
                    # Fill output arrays
                    for key in output_arrays.keys():
                        if key in result:
                            output_arrays[key][i, j] = result[key]
                    
                    result_idx += 1
            
            chunk_idx += 1
    
    # Save results
    if output_file:
        print("Saving results...")
        
        # # Update profile for different data types
        # float_profile = profile.copy()
        # float_profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        
        # int_profile = profile.copy()
        # int_profile.update(dtype=rasterio.int16, count=1, compress='lzw')
        
        # Save all output arrays
        output_files = {}
        # for key, array in output_arrays.items():
        #     output_filename = f"{output_file}_{key}.tif"
        #     current_profile = int_profile if key == 'n_obs' else float_profile
            
        #     with rasterio.open(output_filename, 'w', **current_profile) as dst:
        #         dst.write(array, 1)
            
        #     output_files[key] = output_filename
        #     print(f"  Saved {key} to: {output_filename}")

        # Write COG here of stack
        # Stack
        # move axis of the stack so bands is first
        stack = np.transpose([
                    np.where(np.isnan(output_arrays['trendslope']), nodata_value, output_arrays['trendslope']),    
                    np.where(np.isnan(output_arrays['trendintercept']), nodata_value, output_arrays['trendintercept']),
                    np.where(np.isnan(output_arrays['r2']), nodata_value, output_arrays['r2']), 
                    np.where(np.isnan(output_arrays['trendslope_lo']), nodata_value, output_arrays['trendslope_lo']), 
                    np.where(np.isnan(output_arrays['trendslope_hi']), nodata_value, output_arrays['trendslope_hi']),
                    np.where(np.isnan(output_arrays['kendall_tau']), nodata_value, output_arrays['kendall_tau']),
                    np.where(np.isnan(output_arrays['kendall_pvalue']), nodata_value, output_arrays['kendall_pvalue'])
                              #,output_arrays['n_obs']
                             ],
                             [0,1,2])
        stack_names = ['trendslope', 'trendintercept', 'r2', 
                       'trendslope_lo','trendslope_hi',
                       'kendall_tau','kendall_pvalue'
                       #,'n_obs'
                      ]

        if do_ols: 
            TREND_METHOD = 'ols'
        else:
            TREND_METHOD = 'theilsen'
            
        print(f"Write final {TREND_METHOD} COG...")
        write_cog(
                    stack, 
                  f"{output_file}_{TREND_METHOD}.tif", 
                  crs, 
                  transform,
                  stack_names, 
                  out_crs = crs, 
                  #resolution = (res, res),
                  #align = True, 
                  input_nodata_value=nodata_value,
                  resampling='cubic'
                 )
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    for key, array in output_arrays.items():
        if key == 'n_obs':
            valid_data = array[array > 0]
            if len(valid_data) > 0:
                print(f"{key:12s}: mean={np.mean(valid_data):.1f}, "
                      f"median={np.median(valid_data):.1f}, "
                      f"min={np.min(valid_data)}, max={np.max(valid_data)}")
        else:
            valid_data = array[~np.isnan(array)]
            if len(valid_data) > 0:
                print(f"{key:12s}: mean={np.mean(valid_data):.6f}, "
                      f"median={np.median(valid_data):.6f}, "
                      f"std={np.std(valid_data):.6f}")
    
    # Calculate significance statistics
    if 'kendall_pvalue' in output_arrays:
        p_values = output_arrays['kendall_pvalue']
        valid_p = p_values[~np.isnan(p_values)]
        if len(valid_p) > 0:
            sig_05 = np.sum(valid_p < 0.05) / len(valid_p) * 100
            sig_01 = np.sum(valid_p < 0.01) / len(valid_p) * 100
            print(f"\nSignificance: {sig_05:.1f}% pixels p<0.05, {sig_01:.1f}% pixels p<0.01")
    
    #return output_arrays, output_files if output_file else None
    return output_arrays

def create_summary_plots(output_arrays, output_file=None, SHOW_PLOT=False):
    """
    Create summary plots of the trend analysis results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Trend map
    im1 = axes[0,0].imshow(output_arrays['trendslope'], cmap='BrBG', #cmap='RdBu_r', 
                          #vmin=np.nanpercentile(output_arrays['trend'], 5),
                          #vmax=np.nanpercentile(output_arrays['trend'], 95)
                          vmin=-2, vmax=2
                          )
    axes[0,0].set_title('Theil-Sen Trend (slope)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # P-values
    im2 = axes[0,1].imshow(output_arrays['kendall_pvalue'], cmap='viridis_r', vmin=0, vmax=0.1)
    axes[0,1].set_title('Kendall P-values')
    plt.colorbar(im2, ax=axes[0,1])
    
    # R-squared
    im3 = axes[0,2].imshow(output_arrays['r2'], cmap='plasma', vmin=0, vmax=1)
    axes[0,2].set_title('R-squared')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Number of observations
    im4 = axes[1,0].imshow(output_arrays['n_obs'], cmap='viridis')
    axes[1,0].set_title('Number of Observations')
    plt.colorbar(im4, ax=axes[1,0])
    
    # Confidence interval width
    ci_width = output_arrays['trendslope_hi'] - output_arrays['trendslope_lo']
    im5 = axes[1,1].imshow(ci_width, cmap='magma', 
                          vmin=np.nanpercentile(ci_width, 5),
                          vmax=np.nanpercentile(ci_width, 95))
    axes[1,1].set_title('Confidence Interval Width')
    plt.colorbar(im5, ax=axes[1,1])
    
    # Kendall's tau
    im6 = axes[1,2].imshow(output_arrays['kendall_tau'], cmap='RdBu', vmin=-1, vmax=1)
    axes[1,2].set_title("Kendall's Tau")
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(f"{output_file}_summary_plots.png", dpi=300, bbox_inches='tight')
        print(f"Summary plots saved to: {output_file}_summary_plots.png")
    
    if SHOW_PLOT:
        plt.show()
    
# ##############
# ############ Bayesian
# ##############


# def bayesian_trend_analysis(values, dates, n_samples=1000, target_accept=0.9):
#     """
#     Perform Bayesian linear trend analysis
    
#     Parameters:
#     values: array of observed values
#     dates: array of corresponding dates
#     n_samples: number of MCMC samples
#     target_accept: target acceptance rate for NUTS sampler
    
#     Returns:
#     dict with posterior statistics
#     """
#     results = {
#         'bayes_slope_mean': np.nan,
#         'bayes_slope_std': np.nan,
#         'bayes_slope_hdi_low': np.nan,
#         'bayes_slope_hdi_high': np.nan,
#         'bayes_intercept_mean': np.nan,
#         'bayes_intercept_std': np.nan,
#         'bayes_sigma_mean': np.nan,
#         'bayes_r_hat': np.nan,
#         'bayes_ess': np.nan,
#         'bayes_prob_positive': np.nan,
#         'bayes_prob_negative': np.nan
#     }
    
#     try:
#         # Standardize dates for numerical stability
#         dates_std = (dates - np.mean(dates)) / np.std(dates)
#         values_std = (values - np.mean(values)) / np.std(values)
        
#         with pm.Model() as model:
#             # Priors
#             slope = pm.Normal('slope', mu=0, sigma=2)
#             intercept = pm.Normal('intercept', mu=0, sigma=2)
#             sigma = pm.HalfNormal('sigma', sigma=1)
            
#             # Linear model
#             mu = intercept + slope * dates_std
            
#             # Likelihood
#             y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=values_std)
            
#             # Sample from posterior
#             trace = pm.sample(
#                 draws=n_samples,
#                 tune=max(500, n_samples//2),
#                 target_accept=target_accept,
#                 return_inferencedata=True,
#                 progressbar=False,
#                 random_seed=42
#             )
        
#         # Extract posterior samples
#         posterior = trace.posterior
#         slope_samples = posterior['slope'].values.flatten()
#         intercept_samples = posterior['intercept'].values.flatten()
#         sigma_samples = posterior['sigma'].values.flatten()
        
#         # Convert slope back to original scale
#         slope_original_scale = slope_samples * (np.std(values) / np.std(dates))
#         intercept_original_scale = (intercept_samples * np.std(values) + 
#                                   np.mean(values) - 
#                                   slope_original_scale * np.mean(dates))
        
#         # Compute statistics
#         results['bayes_slope_mean'] = np.mean(slope_original_scale)
#         results['bayes_slope_std'] = np.std(slope_original_scale)
        
#         # HDI (Highest Density Interval) - 95% credible interval
#         slope_hdi = az.hdi(slope_original_scale, hdi_prob=0.95)
#         results['bayes_slope_hdi_low'] = slope_hdi[0]
#         results['bayes_slope_hdi_high'] = slope_hdi[1]
        
#         results['bayes_intercept_mean'] = np.mean(intercept_original_scale)
#         results['bayes_intercept_std'] = np.std(intercept_original_scale)
#         results['bayes_sigma_mean'] = np.mean(sigma_samples) * np.std(values)
        
#         # Diagnostics
#         summary = az.summary(trace, var_names=['slope'])
#         if not summary.empty:
#             results['bayes_r_hat'] = summary.loc['slope', 'r_hat']
#             results['bayes_ess'] = summary.loc['slope', 'ess_bulk']
        
#         # Probability of positive/negative trends
#         results['bayes_prob_positive'] = np.mean(slope_original_scale > 0)
#         results['bayes_prob_negative'] = np.mean(slope_original_scale < 0)
        
#     except Exception as e:
#         # If Bayesian analysis fails, return NaN values
#         print(f"Bayesian analysis failed: {str(e)}")
#         pass
    
#     return results

# def robust_bayesian_trend_analysis(values, dates, n_samples=500):
#     """
#     Robust Bayesian trend analysis using Student's t-distribution
#     More robust to outliers than normal distribution
#     """
#     results = {
#         'robust_bayes_slope_mean': np.nan,
#         'robust_bayes_slope_std': np.nan,
#         'robust_bayes_slope_hdi_low': np.nan,
#         'robust_bayes_slope_hdi_high': np.nan,
#         'robust_bayes_nu_mean': np.nan,  # degrees of freedom parameter
#         'robust_bayes_prob_positive': np.nan
#     }
    
#     try:
#         # Standardize for numerical stability
#         dates_std = (dates - np.mean(dates)) / np.std(dates)
#         values_std = (values - np.mean(values)) / np.std(values)
        
#         with pm.Model() as robust_model:
#             # Priors
#             slope = pm.Normal('slope', mu=0, sigma=2)
#             intercept = pm.Normal('intercept', mu=0, sigma=2)
#             sigma = pm.HalfNormal('sigma', sigma=1)
#             # Degrees of freedom for t-distribution (lower = more robust to outliers)
#             nu = pm.Exponential('nu', lam=1/10) + 1
            
#             # Linear model
#             mu = intercept + slope * dates_std
            
#             # Robust likelihood using Student's t-distribution
#             y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=values_std)
            
#             # Sample
#             trace = pm.sample(
#                 draws=n_samples,
#                 tune=max(500, n_samples//2),
#                 target_accept=0.85,
#                 return_inferencedata=True,
#                 progressbar=False,
#                 random_seed=42
#             )
        
#         # Extract and transform results
#         posterior = trace.posterior
#         slope_samples = posterior['slope'].values.flatten()
#         nu_samples = posterior['nu'].values.flatten()
        
#         # Convert to original scale
#         slope_original_scale = slope_samples * (np.std(values) / np.std(dates))
        
#         results['robust_bayes_slope_mean'] = np.mean(slope_original_scale)
#         results['robust_bayes_slope_std'] = np.std(slope_original_scale)
        
#         slope_hdi = az.hdi(slope_original_scale, hdi_prob=0.95)
#         results['robust_bayes_slope_hdi_low'] = slope_hdi[0]
#         results['robust_bayes_slope_hdi_high'] = slope_hdi[1]
        
#         results['robust_bayes_nu_mean'] = np.mean(nu_samples)
#         results['robust_bayes_prob_positive'] = np.mean(slope_original_scale > 0)
        
#     except Exception as e:
#         print(f"Robust Bayesian analysis failed: {str(e)}")
#         pass
    
#     return results

# def compute_comprehensive_pixel_statistics(args):
#     """
#     Enhanced version with Bayesian methods
#     """
#     pixel_values, pixel_dates, pixel_idx, use_bayesian, use_robust_bayesian = args
    
#     # Initialize results dictionary with all methods
#     results = {
#         # Classical methods
#         'trendslope': np.nan,
#         'trendintercept': np.nan,
#         'r2': np.nan,
#         'kendall_tau': np.nan,
#         'kendall_pvalue': np.nan,
#         'n_obs': 0,
#         'trendslope_lo': np.nan,
#         'trendslope_hi': np.nan,
#         'pixel_idx': pixel_idx,
        
#         # Bayesian methods
#         'bayes_slope_mean': np.nan,
#         'bayes_slope_std': np.nan,
#         'bayes_slope_hdi_low': np.nan,
#         'bayes_slope_hdi_high': np.nan,
#         'bayes_intercept_mean': np.nan,
#         'bayes_intercept_std': np.nan,
#         'bayes_sigma_mean': np.nan,
#         'bayes_r_hat': np.nan,
#         'bayes_ess': np.nan,
#         'bayes_prob_positive': np.nan,
#         'bayes_prob_negative': np.nan,
        
#         # Robust Bayesian methods
#         'robust_bayes_slope_mean': np.nan,
#         'robust_bayes_slope_std': np.nan,
#         'robust_bayes_slope_hdi_low': np.nan,
#         'robust_bayes_slope_hdi_high': np.nan,
#         'robust_bayes_nu_mean': np.nan,
#         'robust_bayes_prob_positive': np.nan
#     }
    
#     # Remove invalid data points
#     valid_mask = ~(np.isnan(pixel_values) | np.isnan(pixel_dates) | 
#                    (pixel_values == -9999) | (pixel_dates == -9999))
    
#     if np.sum(valid_mask) < 3:  # Need at least 3 points for trend
#         return results
    
#     valid_values = pixel_values[valid_mask]
#     valid_dates = pixel_dates[valid_mask]
#     results['n_obs'] = len(valid_values)
    
#     # Classical methods (Theil-Sen, Kendall's tau, etc.)
#     try:
#         # Theil-Sen trend estimation
#         trendslope, trendintercept, trendslope_lo, trendslope_hi = theilslopes(valid_values, valid_dates)
#         results['trendslope'] = trendslope
#         results['trendintercept'] = trendintercept
#         results['trendslope_lo'] = trendslope_lo
#         results['trendslope_hi'] = trendslope_hi
        
#         # Kendall's tau test
#         tau, kendall_p = kendalltau(valid_dates, valid_values)
#         results['kendall_tau'] = tau
#         results['kendall_pvalue'] = kendall_p
        
#         # R-squared calculation
#         predicted_values = slope * valid_dates + intercept
#         if len(valid_values) > 1:
#             correlation, _ = pearsonr(valid_values, predicted_values)
#             results['r2'] = correlation ** 2 if not np.isnan(correlation) else np.nan
            
#     except Exception as e:
#         pass
    
#     # Bayesian methods
#     if use_bayesian and len(valid_values) >= 4:  # Need more points for Bayesian
#         try:
#             bayes_results = bayesian_trend_analysis(valid_values, valid_dates)
#             results.update(bayes_results)
#         except Exception as e:
#             pass
    
#     # Robust Bayesian methods
#     if use_robust_bayesian and len(valid_values) >= 4:
#         try:
#             robust_bayes_results = robust_bayesian_trend_analysis(valid_values, valid_dates)
#             results.update(robust_bayes_results)
#         except Exception as e:
#             pass
    
#     return results

# def process_chunk_with_bayesian(args):
#     """
#     Enhanced chunk processing with Bayesian methods
#     """
#     (chunk_data, chunk_dates, start_row, start_col, chunk_rows, chunk_cols, 
#      use_bayesian, use_robust_bayesian) = args
    
#     results = []
#     for i in range(chunk_rows):
#         for j in range(chunk_cols):
#             pixel_values = chunk_data[:, i, j]
#             pixel_dates = chunk_dates[:, i, j]
#             pixel_idx = ((start_row + i), (start_col + j))
            
#             result = compute_comprehensive_pixel_statistics(
#                 (pixel_values, pixel_dates, pixel_idx, use_bayesian, use_robust_bayesian)
#             )
#             results.append(result)
    
#     return results

# def stack_rasters_and_compute_bayesian_theilsen(value_raster_files, date_raster_files, 
#                                                output_file=None, n_processes=None, 
#                                                chunk_size=200, use_bayesian=True, 
#                                                use_robust_bayesian=True):
#     """
#     Enhanced version with Bayesian trend computation
    
#     Parameters:
#     value_raster_files: list of paths to value rasters
#     date_raster_files: list of paths to corresponding date rasters
#     output_file: base name for output files
#     n_processes: number of processes for parallel computation
#     chunk_size: size of processing chunks (reduced for Bayesian methods)
#     use_bayesian: whether to compute standard Bayesian trends
#     use_robust_bayesian: whether to compute robust Bayesian trends
#     """
    
#     if n_processes is None:
#         n_processes = min(20, cpu_count())  # Reduced for Bayesian computation
    
#     print(f"Using {n_processes} processes for computation")
#     print(f"Bayesian methods enabled: Standard={use_bayesian}, Robust={use_robust_bayesian}")
    
#     # [Previous data loading code remains the same...]
#     # Validate input
#     if len(value_raster_files) != len(date_raster_files):
#         raise ValueError("Number of value and date rasters must match")
    
#     n_rasters = len(value_raster_files)
#     print(f"Processing {n_rasters} raster pairs...")
    
#     # Read the first raster to get metadata
#     with rasterio.open(value_raster_files[0]) as src:
#         profile = src.profile
#         height, width = src.height, src.width
#         transform = src.transform
#         crs = src.crs
    
#     print(f"Raster dimensions: {height} x {width}")
    
#     # Initialize arrays for all data
#     value_stack = np.zeros((n_rasters, height, width), dtype=np.float32)
#     date_stack = np.zeros((n_rasters, height, width), dtype=np.float32)
    
#     # Read all rasters
#     print("Reading value and date rasters...")
#     for i, (value_file, date_file) in enumerate(zip(value_raster_files, date_raster_files)):
#         print(f"  Reading pair {i+1}/{n_rasters}")
        
#         with rasterio.open(value_file) as src:
#             value_stack[i] = src.read(1)
        
#         date_stack[i] = read_date_info(date_file)
    
#     print("Preparing data for parallel processing...")
    
#     # Enhanced output arrays including Bayesian results
#     output_arrays = {
#         # Classical methods
#         'trendslope': np.full((height, width), np.nan, dtype=np.float32),
#         'trendintercept': np.full((height, width), np.nan, dtype=np.float32),
#         'r2': np.full((height, width), np.nan, dtype=np.float32),
#         'kendall_tau': np.full((height, width), np.nan, dtype=np.float32),
#         'kendall_pvalue': np.full((height, width), np.nan, dtype=np.float32),
#         'n_obs': np.full((height, width), 0, dtype=np.int16),
#         'trendslope_lo': np.full((height, width), np.nan, dtype=np.float32),
#         'trendslope_hi': np.full((height, width), np.nan, dtype=np.float32),
#     }
    
#     # Add Bayesian arrays if requested
#     if use_bayesian:
#         bayesian_arrays = {
#             'bayes_slope_mean': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_slope_std': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_slope_hdi_low': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_slope_hdi_high': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_intercept_mean': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_sigma_mean': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_r_hat': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_ess': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_prob_positive': np.full((height, width), np.nan, dtype=np.float32),
#             'bayes_prob_negative': np.full((height, width), np.nan, dtype=np.float32),
#         }
#         output_arrays.update(bayesian_arrays)
    
#     if use_robust_bayesian:
#         robust_bayesian_arrays = {
#             'robust_bayes_slope_mean': np.full((height, width), np.nan, dtype=np.float32),
#             'robust_bayes_slope_std': np.full((height, width), np.nan, dtype=np.float32),
#             'robust_bayes_slope_hdi_low': np.full((height, width), np.nan, dtype=np.float32),
#             'robust_bayes_slope_hdi_high': np.full((height, width), np.nan, dtype=np.float32),
#             'robust_bayes_nu_mean': np.full((height, width), np.nan, dtype=np.float32),
#             'robust_bayes_prob_positive': np.full((height, width), np.nan, dtype=np.float32),
#         }
#         output_arrays.update(robust_bayesian_arrays)
    
#     # Process in chunks
#     chunk_args = []
#     total_chunks = 0
    
#     for start_row in range(0, height, chunk_size):
#         end_row = min(start_row + chunk_size, height)
#         chunk_rows = end_row - start_row
        
#         for start_col in range(0, width, chunk_size):
#             end_col = min(start_col + chunk_size, width)
#             chunk_cols = end_col - start_col
            
#             chunk_values = value_stack[:, start_row:end_row, start_col:end_col]
#             chunk_dates = date_stack[:, start_row:end_row, start_col:end_col]
            
#             chunk_args.append((chunk_values, chunk_dates, start_row, start_col, 
#                              chunk_rows, chunk_cols, use_bayesian, use_robust_bayesian))
#             total_chunks += 1
    
#     print(f"Processing {total_chunks} chunks in parallel...")
    
#     # Process chunks in parallel
#     with Pool(processes=n_processes) as pool:
#         chunk_results = []
#         for i, result in enumerate(pool.imap(process_chunk_with_bayesian, chunk_args)):
#             chunk_results.append(result)
#             if (i + 1) % max(1, total_chunks // 10) == 0:
#                 print(f"  Completed {i+1}/{total_chunks} chunks ({100*(i+1)/total_chunks:.1f}%)")
    
#     # Reassemble results (same as before but with more arrays)
#     print("Assembling results...")
#     chunk_idx = 0
#     for start_row in range(0, height, chunk_size):
#         end_row = min(start_row + chunk_size, height)
        
#         for start_col in range(0, width, chunk_size):
#             end_col = min(start_col + chunk_size, width)
            
#             chunk_result = chunk_results[chunk_idx]
#             result_idx = 0
            
#             for i in range(start_row, end_row):
#                 for j in range(start_col, end_col):
#                     result = chunk_result[result_idx]
                    
#                     for key in output_arrays.keys():
#                         if key in result:
#                             output_arrays[key][i, j] = result[key]
                    
#                     result_idx += 1
            
#             chunk_idx += 1
    
#     # Save results
#     if output_file:
#         print("Saving results...")
        
#         float_profile = profile.copy()
#         float_profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        
#         int_profile = profile.copy()
#         int_profile.update(dtype=rasterio.int16, count=1, compress='lzw')
        
#         output_files = {}
#         for key, array in output_arrays.items():
#             output_filename = f"{output_file}_{key}.tif"
#             current_profile = int_profile if key == 'n_obs' else float_profile
            
#             with rasterio.open(output_filename, 'w', **current_profile) as dst:
#                 dst.write(array, 1)
            
#             output_files[key] = output_filename
#             print(f"  Saved {key} to: {output_filename}")
    
#     # Enhanced summary statistics
#     print("\n" + "="*60)
#     print("COMPREHENSIVE TREND ANALYSIS SUMMARY")
#     print("="*60)
    
#     # Classical methods summary
#     print("\nCLASSICAL METHODS:")
#     for key in ['trendslope', 'kendall_pvalue', 'r2', 'kendall_tau', 'n_obs']:
#         if key in output_arrays:
#             array = output_arrays[key]
#             if key == 'n_obs':
#                 valid_data = array[array > 0]
#             else:
#                 valid_data = array[~np.isnan(array)]
            
#             if len(valid_data) > 0:
#                 print(f"  {key:15s}: mean={np.mean(valid_data):.6f}, "
#                       f"median={np.median(valid_data):.6f}, "
#                       f"std={np.std(valid_data):.6f}")
    
#     # Bayesian methods summary
#     if use_bayesian:
#         print("\nBAYESIAN METHODS:")
#         for key in ['bayes_slope_mean', 'bayes_slope_std', 'bayes_prob_positive', 'bayes_r_hat']:
#             if key in output_arrays:
#                 array = output_arrays[key]
#                 valid_data = array[~np.isnan(array)]
#                 if len(valid_data) > 0:
#                     print(f"  {key:20s}: mean={np.mean(valid_data):.6f}, "
#                           f"median={np.median(valid_data):.6f}")
    
#     # Robust Bayesian methods summary
#     if use_robust_bayesian:
#         print("\nROBUST BAYESIAN METHODS:")
#         for key in ['robust_bayes_slope_mean', 'robust_bayes_nu_mean', 'robust_bayes_prob_positive']:
#             if key in output_arrays:
#                 array = output_arrays[key]
#                 valid_data = array[~np.isnan(array)]
#                 if len(valid_data) > 0:
#                     print(f"  {key:25s}: mean={np.mean(valid_data):.6f}, "
#                           f"median={np.median(valid_data):.6f}")
    
#     return output_arrays, output_files if output_file else None

##########
########## Kendall Trends
##########

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch
from rasterio.plot import show
import os

def classify_kendall_results(tau_array, p_array, alpha=0.05):
    """
    Classify pixels based on Kendall's Tau and p-values
    
    Returns:
    - class_array: Classified array (integer values)
    - mask_array: Boolean array of valid pixels
    """
    # Initialize output arrays
    height, width = tau_array.shape
    class_array = np.zeros((height, width), dtype=np.int8)
    mask_array = ~(np.isnan(tau_array) | np.isnan(p_array))
    
    # Define categories (considering both strength and significance)
    # 0: No data/invalid pixels
    # 1: Strong significant positive (tau >= 0.7, p < alpha)
    # 2: Moderate significant positive (0.4 <= tau < 0.7, p < alpha)
    # 3: Weak significant positive (0.2 <= tau < 0.4, p < alpha)
    # 4: Very weak significant positive (0 < tau < 0.2, p < alpha)
    # 5: Non-significant positive (tau > 0, p >= alpha)
    # 6: Non-significant negative (tau < 0, p >= alpha)
    # 7: Very weak significant negative (-0.2 < tau < 0, p < alpha)
    # 8: Weak significant negative (-0.4 < tau <= -0.2, p < alpha)
    # 9: Moderate significant negative (-0.7 < tau <= -0.4, p < alpha)
    # 10: Strong significant negative (tau <= -0.7, p < alpha)
    
    # Set all valid pixels initially to non-significant classes
    valid_indices = np.where(mask_array)
    
    # Default to non-significant classes
    class_array[valid_indices] = np.where(
        tau_array[valid_indices] >= 0, 
        5,  # Non-significant positive
        6   # Non-significant negative
    )
    
    # Significant positive trends (from strongest to weakest)
    sig_pos = (tau_array >= 0) & (p_array < alpha) & mask_array
    class_array[(tau_array >= 0.7) & sig_pos] = 1  # Strong significant positive
    class_array[(tau_array >= 0.4) & (tau_array < 0.7) & sig_pos] = 2  # Moderate significant positive
    class_array[(tau_array >= 0.2) & (tau_array < 0.4) & sig_pos] = 3  # Weak significant positive
    class_array[(tau_array > 0) & (tau_array < 0.2) & sig_pos] = 4  # Very weak significant positive
    
    # Significant negative trends (from weakest to strongest)
    sig_neg = (tau_array < 0) & (p_array < alpha) & mask_array
    class_array[(tau_array > -0.2) & (tau_array < 0) & sig_neg] = 7  # Very weak significant negative
    class_array[(tau_array <= -0.2) & (tau_array > -0.4) & sig_neg] = 8  # Weak significant negative
    class_array[(tau_array <= -0.4) & (tau_array > -0.7) & sig_neg] = 9  # Moderate significant negative
    class_array[(tau_array <= -0.7) & sig_neg] = 10  # Strong significant negative
    
    return class_array, mask_array

def save_classified_kendall_raster(class_array, reference_raster_path, output_path):
    """
    Save classified Kendall's Tau results as a raster
    """
    with rasterio.open(reference_raster_path) as src:
        profile = src.profile
    
    # Update profile for the classified raster
    profile.update(
        dtype=rasterio.int8,
        count=1,
        compress='lzw',
        nodata=0
    )
    
    # Save the classified raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(class_array, 1)
    
    print(f"Classified Kendall's Tau raster saved to: {output_path}")

# Color map for different classes (red=negative, blue=positive)
cmap_colors_kendall_classes = {
    0: (1.0, 1.0, 1.0, 0.0),  # No data - transparent
    1: (0.0, 0.0, 0.8, 1.0),  # Strong sig. positive - dark blue
    2: (0.3, 0.3, 0.9, 1.0),  # Moderate sig. positive - medium blue
    3: (0.5, 0.5, 1.0, 1.0),  # Weak sig. positive - light blue
    4: (0.7, 0.7, 1.0, 1.0),  # Very weak sig. positive - very light blue
    5: (0.8, 0.8, 0.8, 1.0),  # Non-sig. positive - light gray
    6: (0.6, 0.6, 0.6, 1.0),  # Non-sig. negative - dark gray
    7: (1.0, 0.7, 0.7, 1.0),  # Very weak sig. negative - very light red
    8: (1.0, 0.5, 0.5, 1.0),  # Weak sig. negative - light red
    9: (0.9, 0.3, 0.3, 1.0),  # Moderate sig. negative - medium red
    10: (0.8, 0.0, 0.0, 1.0)  # Strong sig. negative - dark red
}
class_labels_kendall_classes = {
            1: "Strong sig. [+]",
            2: "Mod. sig. [+]",
            3: "Weak sig. [+]",
            4: "Very weak sig. [+]",
            5: "Non-sig. [+]",
            6: "Non-sig. [-]",
            7: "Very weak sig. [-]",
            8: "Weak sig. [-]",
            9: "Mod. sig. [-]",
            10: "Strong sig. [-]"
        }
    
def plot_classified_kendall_map(class_array, output_path=None, figsize=(12, 12), 
                              dpi=300, class_labels=class_labels_kendall_classes, cmap_colors=cmap_colors_kendall_classes):
    """
    Create a classified map visualization of Kendall's Tau results
    """
    # Define default class labels if not provided
    if class_labels is None:
        class_labels = class_labels_kendall_classes
    
    # Rest of the function continues as before...
    
    # Create custom colormap
    colors_list = [cmap_colors[i] for i in range(11)]
    custom_cmap = colors.ListedColormap(colors_list)
    bounds = np.arange(-0.5, 11.5, 1)
    norm = colors.BoundaryNorm(bounds, len(bounds)-1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(class_array, cmap=custom_cmap, norm=norm)
    
    # Add legend
    legend_elements = []
    for class_id, label in class_labels.items():
        legend_elements.append(
            Patch(facecolor=cmap_colors[class_id], label=label)
        )
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
              framealpha=0.7, title="Trend Classes")
    
    ax.set_title("Classified Kendall's Tau Trend Analysis", fontsize=14)
    ax.set_axis_off()
    
    # Add scale bar and north arrow here if needed
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Classified map saved to: {output_path}")
    
    plt.show()
    return fig, ax

def create_summary_statistics_figure(tau_array, p_array, class_array, output_path=None, alpha=0.05):
    """
    Create a summary statistics figure with histograms and pie charts
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(tau_array) | np.isnan(p_array))
    valid_tau = tau_array[valid_mask]
    valid_p = p_array[valid_mask]
    valid_classes = class_array[valid_mask]
    
    if len(valid_tau) == 0:
        print("No valid Kendall's Tau results found")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    
    # 1. Histogram of Kendall's Tau values
    axes[0, 0].hist(valid_tau, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='red', linestyle='--')
    axes[0, 0].set_title("Distribution of Kendall's Tau Values")
    axes[0, 0].set_xlabel("Kendall's Tau")
    axes[0, 0].set_ylabel("Frequency")
    
    # 2. Histogram of p-values
    axes[0, 1].hist(valid_p, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=alpha, color='red', linestyle='--')
    axes[0, 1].set_title("Distribution of p-values")
    axes[0, 1].set_xlabel("p-value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].text(alpha*1.1, axes[0, 1].get_ylim()[1]*0.9, f'α={alpha}', 
                  color='red', fontsize=12)
    
    # 3. Histogram of classes
    axes[0, 2].hist(valid_classes, bins=10, color='black', edgecolor='black', alpha=0.7)
    # axes[0, 2].axvline(x=alpha, color='red', linestyle='--')
    axes[0, 2].set_title("Distribution of Kendall's tau classes")
    axes[0, 2].set_xlabel("class")
    axes[0, 2].set_ylabel("Frequency")
    # axes[0, 2].text(alpha*1.1, axes[0, 1].get_ylim()[1]*0.9, f'α={alpha}', 
    #               color='red', fontsize=12)
    
    # 4. Pie chart of trend directions
    positive_trends = np.sum(valid_tau > 0)
    negative_trends = np.sum(valid_tau < 0)
    no_trends = np.sum(valid_tau == 0)
    
    direction_labels = ['Positive', 'Negative', 'No trend']
    direction_sizes = [positive_trends, negative_trends, no_trends]
    direction_colors = ['steelblue', 'indianred', 'lightgray']
    
    axes[1, 0].pie(direction_sizes, labels=direction_labels, colors=direction_colors,
                 autopct='%1.1f%%', startangle=90, shadow=False)
    axes[1, 0].set_title('Trend Directions')
    
    # 5. Pie chart of significance and strength
    # Calculate categories
    abs_tau = np.abs(valid_tau)
    strong_sig = np.sum((abs_tau >= 0.7) & (valid_p < alpha))
    moderate_sig = np.sum((abs_tau >= 0.4) & (abs_tau < 0.7) & (valid_p < alpha))
    weak_sig = np.sum((abs_tau >= 0.2) & (abs_tau < 0.4) & (valid_p < alpha))
    very_weak_sig = np.sum((abs_tau < 0.2) & (valid_p < alpha))
    non_sig = np.sum(valid_p >= alpha)
    
    strength_labels = ['Strong sig.', 'Moderate sig.', 'Weak sig.', 
                       'Very weak sig.', 'Non-significant']
    strength_sizes = [strong_sig, moderate_sig, weak_sig, very_weak_sig, non_sig]
    strength_colors = ['darkblue', 'royalblue', 'skyblue', 'lightblue', 'lightgray']
    
    # Filter out zero values for better visualization
    filtered_labels = [l for l, s in zip(strength_labels, strength_sizes) if s > 0]
    filtered_sizes = [s for s in strength_sizes if s > 0]
    filtered_colors = [c for c, s in zip(strength_colors, strength_sizes) if s > 0]
    
    if filtered_sizes:  # Check if there are any non-zero values
        axes[1, 1].pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors,
                     autopct='%1.1f%%', startangle=90, shadow=False)
    axes[1, 1].set_title('Trend Strength and Significance')

    # 6. Pie chart of classes from combination of significance and strength
    # Calculate categories
    abs_tau = np.abs(valid_tau)
    strong_sig = np.sum((abs_tau >= 0.7) & (valid_p < alpha))
    moderate_sig = np.sum((abs_tau >= 0.4) & (abs_tau < 0.7) & (valid_p < alpha))
    weak_sig = np.sum((abs_tau >= 0.2) & (abs_tau < 0.4) & (valid_p < alpha))
    very_weak_sig = np.sum((abs_tau < 0.2) & (valid_p < alpha))
    non_sig = np.sum(valid_p >= alpha)
    
    strength_labels = ['Strong sig.', 'Moderate sig.', 'Weak sig.', 
                       'Very weak sig.', 'Non-significant']
    strength_sizes = [strong_sig, moderate_sig, weak_sig, very_weak_sig, non_sig]
    strength_colors = ['darkblue', 'royalblue', 'skyblue', 'lightblue', 'lightgray']
    
    # Filter out zero values for better visualization
    filtered_labels = [l for l, s in zip(list(class_labels_kendall_classes.values()), valid_classes) if s > 0]
    filtered_classes = [s for s in valid_classes if s > 0]
    filtered_colors = [c for c, s in zip(list(cmap_colors_kendall_classes.values()), strength_sizes) if s > 0]

    # kendall_classes_labels = list(class_labels_kendall_classes.values())
    # filtered_valid_classes = [s for s in valid_classes if s > 0]
    # kendall_classes_colors = list(cmap_colors_kendall_classes.values())
    if False:
        if filtered_sizes:  # Check if there are any non-zero values
            axes[1, 2].pie(filtered_classes, labels=filtered_labels, colors=filtered_colors,
                         autopct='%1.1f%%', startangle=90, shadow=False)
        axes[1, 2].set_title('Trend Classes')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Summary statistics figure saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("KENDALL'S TAU SUMMARY STATISTICS")
    print("="*50)
    print(f"Total valid pixels: {len(valid_tau)}")
    print(f"Mean τ: {np.mean(valid_tau):.4f}")
    print(f"Median τ: {np.median(valid_tau):.4f}")
    print(f"Range: [{np.min(valid_tau):.4f}, {np.max(valid_tau):.4f}]")
    print(f"Significant pixels (p<{alpha}): {np.sum(valid_p < alpha)} ({100*np.sum(valid_p < alpha)/len(valid_p):.1f}%)")
    
    return fig

import datetime
import numpy as np
from osgeo import gdal, gdal_array

def create_kendall_class_raster(class_array, reference_raster_path, output_path, class_labels):
    """
    Create a kendall's tau class raster for interpreting theil-sens trends
    with embedded color table for GIS software
    
    Parameters:
    class_array: numpy array with classified values
    reference_raster_path: path to a reference raster for spatial metadata
    output_path: path to save the output classes raster
    class_labels: dictionary mapping class values to descriptive labels
    """
    # Define colors for each class (in RGBA)
    class_colors = {
        0: (255, 255, 255, 0),     # No data - transparent
        1: (0, 0, 204, 255),       # Strong sig. positive - dark blue
        2: (51, 51, 255, 255),     # Moderate sig. positive - medium blue
        3: (102, 102, 255, 255),   # Weak sig. positive - light blue
        4: (153, 153, 255, 255),   # Very weak sig. positive - very light blue
        5: (204, 204, 204, 255),   # Non-sig. positive - light gray
        6: (153, 153, 153, 255),   # Non-sig. negative - dark gray
        7: (255, 153, 153, 255),   # Very weak sig. negative - very light red
        8: (255, 102, 102, 255),   # Weak sig. negative - light red
        9: (255, 51, 51, 255),     # Moderate sig. negative - medium red
        10: (204, 0, 0, 255)       # Strong sig. negative - dark red
    }
    
    # Open reference raster to get geotransform and projection
    with rasterio.open(reference_raster_path) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs
        #ndv = src.nodata
    
    # Create a copy of the input array to ensure data type is int8
    output_array = class_array.astype(np.int8)

    # Write COG here of stack
    # Stack
    # move axis of the stack so bands is first
    stack = np.transpose([output_array], [0,1,2])
    stack_names = ['kendallclasses']
    
    print("Write final theilsen COG...")
    write_cog(
            stack, 
          output_path, 
          crs, 
          transform,
          stack_names, 
          out_crs = crs, 
          #resolution = (res, res),
          #align = True, 
          input_nodata_value=0,
          resampling='cubic'
         )
    
    # Close the dataset to finalize the file
    band = None
    dataset = None
    
    print(f"Classified Kendall's Tau raster saved to: {output_path}")
    print(f"  with color table and {len(class_labels)} class definitions")
    
def save_classified_kendall_raster(class_array, reference_raster_path, output_path, class_labels):
    """
    Save classified Kendall's Tau results as a raster with class labels as metadata
    
    Parameters:
    class_array: numpy array with classified values
    reference_raster_path: path to a reference raster for spatial metadata
    output_path: path to save the output classified raster
    class_labels: dictionary mapping class values to descriptive labels
    """
    with rasterio.open(reference_raster_path) as src:
        profile = src.profile
    
    # Update profile for the classified raster
    profile.update(
        dtype=rasterio.int8,
        count=1,
        compress='lzw',
        nodata=0
    )
    
    # Save the classified raster with metadata
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(class_array, 1)
        
        # Add band description using class labels
        dst.set_band_description(1, "Kendall's Tau Classification")
        
        # Add class definitions as metadata
        class_info = {}
        for class_id, label in class_labels.items():
            class_info[f"CLASS_{class_id}"] = label
        
        # Add general metadata
        metadata = {
            'description': "Kendall's Tau trend classification",
            'created_by': "Kendall's Tau trend analysis script",
            'creation_date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'nodata_value': '0',
            'classification_info': "Classes represent trend direction, strength, and significance",
            **class_info  # Add all class definitions
        }
        
        # Write metadata
        dst.update_tags(**metadata)
        
    print(f"Classified Kendall's Tau raster saved to: {output_path}")
    print(f"  with {len(class_labels)} class definitions added to metadata")

def analyze_and_visualize_kendall_results(tau_array, p_array, reference_raster_path, 
                                         output_dir="kendall_results", alpha=0.05):
    """
    Complete pipeline to analyze, classify, save, and visualize Kendall's Tau results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define class labels
    class_labels = {
        1: "Strong sig. positive",
        2: "Moderate sig. positive",
        3: "Weak sig. positive",
        4: "Very weak sig. positive",
        5: "Non-sig. positive",
        6: "Non-sig. negative",
        7: "Very weak sig. negative",
        8: "Weak sig. negative",
        9: "Moderate sig. negative",
        10: "Strong sig. negative"
    }

    out_file_stem = os.path.basename(reference_raster_path).split('.tif')[0]
    
    # 1. Classify Kendall's Tau results
    print("Classifying Kendall's Tau results...")
    class_array, mask_array = classify_kendall_results(tau_array, p_array, alpha)
    
    # # 2. Save classified raster with class labels as metadata
    # classified_output_path = os.path.join(output_dir, out_file_stem + "_kendall_tau_classified.tif")
    # save_classified_kendall_raster(class_array, reference_raster_path, classified_output_path, class_labels)
    
    # 3. Create and save categorical/class raster with class values and color table
    classes_output_path = os.path.join(output_dir, out_file_stem + "_kendallclasses.tif")
    create_kendall_class_raster(class_array, reference_raster_path, classes_output_path, class_labels)
    
    # 4. Create and save classified map
    map_output_path = os.path.join(output_dir, out_file_stem + "_kendall_tau_classified_map.png")
    print("Creating classified map visualization...")
    plot_classified_kendall_map(class_array, map_output_path, class_labels=class_labels_kendall_classes)
    
    # 5. Create and save summary statistics figure
    stats_output_path = os.path.join(output_dir, out_file_stem + "_kendall_tau_summary_stats.png")
    print("Creating summary statistics visualization...")
    create_summary_statistics_figure(tau_array, p_array, class_array, stats_output_path, alpha)
    
    # 6. Save continuous Kendall's Tau and p-value rasters for reference
    with rasterio.open(reference_raster_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        
    ## These are already returned from the theil-sen run
    # tau_output_path = os.path.join(output_dir, out_file_stem + "_kendall_tau_values.tif")
    # with rasterio.open(tau_output_path, 'w', **profile) as dst:
    #     dst.write(tau_array, 1)
    #     dst.set_band_description(1, "Kendall's Tau values")
    #     dst.update_tags(description="Kendall's Tau correlation values")
    
    # pval_output_path = os.path.join(output_dir, out_file_stem + "_kendall_pvalues.tif")
    # with rasterio.open(pval_output_path, 'w', **profile) as dst:
    #     dst.write(p_array, 1)
    #     dst.set_band_description(1, "Kendall's Tau p-values")
    #     dst.update_tags(description="Statistical significance (p-values)", 
    #                    alpha_threshold=str(alpha))
    
    print(f"Analysis complete. All results saved to directory: {output_dir}")
    
    # Return paths to all outputs
    return {
        #"classified_raster": classified_output_path,
        "classes_raster": classes_output_path,
        "classified_map": map_output_path,
        "summary_stats": stats_output_path,
        #"tau_values": tau_output_path,
        #"p_values": pval_output_path
    }
def suggest_alpha_by_temporal_resolution(n_time_points, data_type="satellite"):
    """
    Suggest alpha thresholds based on temporal characteristics
    """
    suggestions = {}
    
    if n_time_points < 10:
        suggestions['conservative'] = 0.10
        suggestions['standard'] = 0.15
        suggestions['liberal'] = 0.20
        suggestions['note'] = "Few time points - consider more liberal alpha"
        
    elif n_time_points < 20:
        suggestions['conservative'] = 0.05
        suggestions['standard'] = 0.10
        suggestions['liberal'] = 0.15
        suggestions['note'] = "Moderate time series - standard approaches work"
        
    else:  # >= 20 time points
        suggestions['conservative'] = 0.01
        suggestions['standard'] = 0.05
        suggestions['liberal'] = 0.10
        suggestions['note'] = "Long time series - can use stricter alpha"
    
    # Adjust for data type
    if data_type == "satellite":
        suggestions['note'] += ". Consider noise and cloud contamination."
    elif data_type == "climate":
        suggestions['note'] += ". Climate data often has strong autocorrelation."
    
    return suggestions

def wrapper_kendall_tau_analysis(DICT_run, DATE_RASTER_FILES, N_PROCESSES, CHUNK_SIZE, ALPHA=0.1, DO_OLS=True, DETECT_BREAKS=True):
    
    # Define your raster files
    VALUE_RASTER_FILES = DICT_run['value_raster_files'] 
    OUT_FILE = DICT_run['out_file']
    OUTDIR = DICT_run['outdir']
    #!mkdir -p $outdir

    # Run the main trend analysis
    output_arrays = stack_rasters_and_compute_trend(
        value_raster_files=VALUE_RASTER_FILES,
        date_raster_files=DATE_RASTER_FILES,
        output_file=f"{OUTDIR}/{OUT_FILE}",
        n_processes=N_PROCESSES,
        chunk_size=CHUNK_SIZE,
        do_ols=DO_OLS
    )
    
    if DO_OLS: 
        TREND_METHOD = 'ols'
    else:
        TREND_METHOD = 'theilsen'

    create_summary_plots(output_arrays, f"{OUTDIR}/{OUT_FILE}_{TREND_METHOD}")
    
    # Then run the analysis and visualization
    result_paths = analyze_and_visualize_kendall_results(
        output_arrays['kendall_tau'],
        output_arrays['kendall_pvalue'],
        reference_raster_path=f"{OUTDIR}/{OUT_FILE}_{TREND_METHOD}.tif",
        output_dir=OUTDIR,
        alpha=ALPHA
    )
    
    # ADDITION: Breakpoint detection with multiprocessing
    if DETECT_BREAKS:
        breakpoint_file = detect_breakpoints_raster(
            value_raster_files=VALUE_RASTER_FILES,
            DATE_RASTER_FILES=DATE_RASTER_FILES,
            OUT_FILE=f"{OUTDIR}/{OUT_FILE}",
            TREND_METHOD=TREND_METHOD,
            chunk_size=CHUNK_SIZE,
            min_break_position=0.7,  # Only detect breaks in last 30% of time series
            penalty=10,  # Adjust based on desired sensitivity
            n_processes=N_PROCESSES  # Use same number of processes as trend analysis
        )
        
        # Add to result paths
        result_paths['breakpoints'] = breakpoint_file
    
    return result_paths

import argparse
import sys
import os
from pathlib import Path
import json

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run trend analysis on raster time series data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_trends.py --value-rasters data/ndvi_*.tif --date-rasters data/dates_*.tif --output results/trend.tif --outdir results/
  python compute_trends.py --config config.json --alpha 0.05 --do-ols --no-breakpoints
        """
    )
    
    # Input data arguments
    parser.add_argument(
        '--value-rasters', '-v',
        nargs='+',
        help='List of value raster files (e.g., NDVI, biomass) in chronological order'
    )
    
    parser.add_argument(
        '--date-rasters', '-d', 
        nargs='+',
        help='List of date raster files corresponding to value rasters'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='JSON configuration file with input parameters'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output filename (without extension)'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='./results',
        help='Output directory (default: ./results)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Significance level for trend analysis (default: 0.1)'
    )
    
    parser.add_argument(
        '--do-ols',
        action='store_true',
        help='Use OLS regression analysis (default: Theil-Sen)'
    )
    
    parser.add_argument(
        '--no-breakpoints',
        action='store_true', 
        help='Skip breakpoint detection analysis'
    )
    
    # Processing parameters
    parser.add_argument(
        '--n-processes',
        type=int,
        default=30,
        help='Number of processes for parallel processing (default: 30)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=300,
        help='Chunk size for processing (default: 300)'
    )
    
    parser.add_argument(
        '--min-break-position',
        type=float,
        default=0.7,
        help='Minimum break position for breakpoint detection (default: 0.7)'
    )
    
    parser.add_argument(
        '--penalty',
        type=float,
        default=10,
        help='Penalty parameter for breakpoint detection (default: 10)'
    )
    
    # Utility arguments
    parser.add_argument(
        '--verbose', '-V',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os
import sys
from urllib.parse import urlparse

def is_s3_path(path):
    """Check if a path is an S3 URL."""
    return path.startswith('s3://')

def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key

def check_s3_object_exists(s3_client, bucket, key):
    """Check if an S3 object exists."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            # Other error (permissions, etc.)
            raise e

def validate_s3_credentials():
    """Validate S3 credentials and access."""
    try:
        s3_client = boto3.client('s3')
        # Test credentials with a simple operation
        s3_client.list_buckets()
        return s3_client
    except NoCredentialsError:
        raise Exception("AWS credentials not found. Please configure AWS credentials.")
    except ClientError as e:
        raise Exception(f"AWS credentials error: {e}")
    except Exception as e:
        raise Exception(f"Error connecting to S3: {e}")

def validate_inputs(args):
    """Validate input arguments with S3 support."""
    errors = []
    s3_client = None
    
    # Check if either config or direct inputs are provided
    if not args.config and not args.value_rasters:
        errors.append("Either --config or --value-rasters must be provided")
    
    # If using direct inputs, check required arguments
    if args.value_rasters:
        if not args.date_rasters:
            errors.append("--date-rasters required when using --value-rasters")
        elif len(args.value_rasters) != len(args.date_rasters):
            errors.append("Number of value rasters must match number of date rasters")
    
    # Check if S3 credentials are needed
    s3_paths_exist = False
    if args.value_rasters:
        s3_paths_exist = any(is_s3_path(path) for path in args.value_rasters)
    if args.date_rasters:
        s3_paths_exist = s3_paths_exist or any(is_s3_path(path) for path in args.date_rasters)
    if args.config and is_s3_path(args.config):
        s3_paths_exist = True
    
    # Initialize S3 client if needed
    if s3_paths_exist:
        try:
            s3_client = validate_s3_credentials()
        except Exception as e:
            errors.append(str(e))
            # Return early if S3 credentials are invalid
            if errors:
                print("Input validation errors:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
    
    # Check file existence for value rasters
    if args.value_rasters:
        for raster in args.value_rasters:
            if is_s3_path(raster):
                try:
                    bucket, key = parse_s3_path(raster)
                    if not check_s3_object_exists(s3_client, bucket, key):
                        errors.append(f"S3 object not found: {raster}")
                except Exception as e:
                    errors.append(f"Error checking S3 object {raster}: {e}")
            else:
                if not os.path.exists(raster):
                    errors.append(f"Local file not found: {raster}")
    
    # Check file existence for date rasters
    if args.date_rasters:
        for raster in args.date_rasters:
            if is_s3_path(raster):
                try:
                    bucket, key = parse_s3_path(raster)
                    if not check_s3_object_exists(s3_client, bucket, key):
                        errors.append(f"S3 object not found: {raster}")
                except Exception as e:
                    errors.append(f"Error checking S3 object {raster}: {e}")
            else:
                if not os.path.exists(raster):
                    errors.append(f"Local file not found: {raster}")
    
    # Check config file existence
    if args.config:
        if is_s3_path(args.config):
            try:
                bucket, key = parse_s3_path(args.config)
                if not check_s3_object_exists(s3_client, bucket, key):
                    errors.append(f"S3 config file not found: {args.config}")
            except Exception as e:
                errors.append(f"Error checking S3 config file {args.config}: {e}")
        else:
            if not os.path.exists(args.config):
                errors.append(f"Local config file not found: {args.config}")
    
    # Check parameter ranges
    if not 0 < args.alpha < 1:
        errors.append("Alpha must be between 0 and 1")
    
    if args.n_processes < 1:
        errors.append("Number of processes must be >= 1")
    
    if args.chunk_size < 1:
        errors.append("Chunk size must be >= 1")
    
    if not 0 < args.min_break_position < 1:
        errors.append("Min break position must be between 0 and 1")
    
    if errors:
        print("Input validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

# Optional: Update load_config function to support S3
def load_config(config_file):
    """Load configuration from JSON file (local or S3)."""
    try:
        if is_s3_path(config_file):
            # Load from S3
            s3_client = boto3.client('s3')
            bucket, key = parse_s3_path(config_file)
            
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            config = json.loads(content)
        else:
            # Load from local file
            with open(config_file, 'r') as f:
                config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1)

def prepare_run_dict(args, config=None):
    """Prepare the DICT_run parameter for wrapper_kendall_tau_analysis."""
    
    if config:
        # Use config file parameters
        DICT_run = {
            'value_raster_files': config.get('value_raster_files', []),
            'out_file': config.get('out_file', args.output),
            'outdir': config.get('outdir', args.outdir)
        }
        
        # Override with command line arguments if provided
        if args.output:
            DICT_run['out_file'] = args.output
        if args.outdir != './results':
            DICT_run['outdir'] = args.outdir
            
    else:
        # Use command line arguments
        DICT_run = {
            'value_raster_files': args.value_rasters,
            'out_file': args.output,
            'outdir': args.outdir
        }
    
    return DICT_run

def main():

    """
    Main script for running trend analysis on raster data.
    
    This script performs trend analysis either theil-sen or OLS, using Kendall's Tau correlation coefficient
    on time series raster data, with options for OLS regression and breakpoint detection.
    """
    
    """Main function."""
    args = parse_arguments()
    
    # Validate inputs
    #validate_inputs(args)
    
    if args.verbose:
        print("Starting trend analysis...")
        print(f"Output directory: {args.outdir}")
        print(f"Output file: {args.output}")
        print(f"Alpha level: {args.alpha}")
        print(f"OLS analysis: {args.do_ols}")
        print(f"Breakpoint detection: {not args.no_breakpoints}")
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
        if args.verbose:
            print(f"Loaded configuration from: {args.config}")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Prepare parameters
    DICT_run = prepare_run_dict(args, config)
    
    # Prepare date raster files
    if config and 'date_raster_files' in config:
        DATE_RASTER_FILES = config['date_raster_files']
    else:
        DATE_RASTER_FILES = args.date_rasters
    
    if args.verbose:
        print(f"Processing {len(DICT_run['value_raster_files'])} raster files...")
        print(f"Using {args.n_processes} processes with chunk size {args.chunk_size}")
    
    try:
        
        # Run the analysis
        result_paths = wrapper_kendall_tau_analysis(
            DICT_run,
            DATE_RASTER_FILES,
            args.n_processes,
            args.chunk_size,
            ALPHA=args.alpha,
            DO_OLS=args.do_ols,
            DETECT_BREAKS=not args.no_breakpoints
        )
        
        if args.verbose:
            print("Analysis completed successfully!")
            print("Output files:")
            for key, path in result_paths.items():
                print(f"  {key}: {path}")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
if __name__ == "__main__":
    sys.exit(main())