import os
import numpy as np
import rasterio
from rasterio.mask import mask
import pandas as pd
import geopandas as gpd
from scipy import stats
import json

import random
import matplotlib.pyplot as plt
from shapely.geometry import box

from rastoolslib import *
import mosaiclib
from CovariateUtils import write_cog
import argparse

def process_multiple_rasters(TILE_NUM_LIST, num_simulations=50, n_cores=None, RANDOM_SEED=True):
    """
    Process multiple rasters in parallel, combining their results.
    
    Parameters:
    -----------
    TILE_NUM_LIST : list
        List of tiles to raster files
    num_simulations : int
        Number of Monte Carlo simulations to run
    n_cores : int
        Number of cores to use (defaults to all available cores - 1)
        
    Returns:
    --------
    dict
        Dictionary with combined results
    """
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)
    
    print(f"Processing {len(TILE_NUM_LIST)} tiles using {n_cores} cores...")
    
    # Create a pool of worker processes
    pool = mp.Pool(processes=n_cores)

    if RANDOM_SEED:
        print('Use random seeds...')
        random_seeds = np.random.randint(0, 10000, size=len(TILE_NUM_LIST))
    else:
        print('Use the same seeds for reproducibility...')
        random_seeds = [100 for t in TILE_NUM_LIST]
    
    # Set up the parallel execution with different seeds for each process
    process_func = partial(CARBON_ACC_ANALYSIS_SINGLE, num_simulations=num_simulations)
    tasks = [(path, seed) for path, seed in zip(TILE_NUM_LIST, random_seeds)]
    
    # Execute processing in parallel with progress bar
    results = []
    with tqdm(total=len(tasks), desc="Processing rasters") as pbar:
        for task in tasks:
            result = pool.apply_async(process_func, args=(task[0],), 
                                     kwds={'random_seed': task[1]},
                                     callback=lambda _: pbar.update())
            results.append(result)
        
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
    
    # Collect results
    raster_results = [res.get() for res in results]
    
    return {
        'raster_results': raster_results,
        # 'combined_mean_sum': combined_mean_sum,
        # 'combined_std': combined_std,
        # 'combined_ci_lower': combined_ci_lower,
        # 'combined_ci_upper': combined_ci_upper,
        # 'combined_pi_lower': combined_pi_lower,
        # 'combined_pi_upper': combined_pi_upper,
        # 'all_simulation_sums': all_simulation_sums
    }

def read_and_prepare_rasters(raster_paths, nodata_value=-9999):
    """
    Read multiple raster datasets and prepare them for analysis by masking nodata values.
    
    Parameters:
    -----------
    raster_paths : list
        List of paths to raster files
    nodata_values : int
        raster nodata value 
        
    Returns:
    --------
    dict
        Dictionary with raster data and metadata
    """
   
    rasters = {}
    
    # Read biomass raster (index 0)
    with rasterio.open(raster_paths[0]) as src:
        rasters['biomass_mean'] = src.read(1)
        rasters['biomass_std'] = src.read(2)
        rasters['meta'] = src.meta
        rasters['transform'] = src.transform
        rasters['crs'] = src.crs
        rasters['shape'] = rasters['biomass_mean'].shape
        
        # Create a mask for valid data (not -9999)
        mask_array = rasters['biomass_mean'] != nodata_value
        
        # Apply mask to biomass data
        rasters['biomass_mean'] = np.where(mask_array, rasters['biomass_mean'], np.nan)
        rasters['biomass_std'] = np.where(mask_array, rasters['biomass_std'], np.nan)
    
    # Read age raster (index 1)
    with rasterio.open(raster_paths[1]) as src:
        rasters['age_mean'] = src.read(1)
        rasters['age_std'] = src.read(2)
        
        # Apply mask to age data
        mask_array = rasters['age_mean'] != nodata_value
        rasters['age_mean'] = np.where(mask_array, rasters['age_mean'], np.nan)
        rasters['age_std'] = np.where(mask_array, rasters['age_std'], np.nan)
    
    # Read landcover raster (index 2)
    with rasterio.open(raster_paths[2]) as src:
        rasters['landcover'] = src.read(1)
        mask_array = rasters['landcover'] != nodata_value
        rasters['landcover'] = np.where(mask_array, rasters['landcover'], np.nan)
    
    # Read topography raster (index 3)
    with rasterio.open(raster_paths[3]) as src:
        rasters['elevation'] = src.read(1)
        rasters['slope'] = src.read(2)
        rasters['tsri'] = src.read(3)
        rasters['tpi'] = src.read(4)
        rasters['slopemask'] = src.read(5)
        
        # Apply mask
        for band in ['elevation', 'slope', 'tsri', 'tpi', 'slopemask']:
            mask_array = rasters[band] != nodata_value
            rasters[band] = np.where(mask_array, rasters[band], np.nan)
            
    if raster_paths[4] is not None:
        # Read tree canopy cover raster (index 4)
        with rasterio.open(raster_paths[4]) as src:
            rasters['canopy_cover'] = src.read(1)
            mask_array = (rasters['canopy_cover'] != nodata_value) & (rasters['canopy_cover'] != 255)
            rasters['canopy_cover'] = np.where(mask_array, rasters['canopy_cover'], np.nan)
    else:
        rasters['canopy_cover'] = np.full_like(rasters['biomass_mean'] , nodata_value)

    if raster_paths[5] is not None:
        # Read tree canopy cover trends raster (index 5)
        with rasterio.open(raster_paths[5]) as src:
            rasters['canopy_trend'] = src.read(1)
            mask_array = (rasters['canopy_trend'] != nodata_value) & (rasters['canopy_trend'] != 255)
            rasters['canopy_trend'] = np.where(mask_array, rasters['canopy_trend'], np.nan)
        
        # Read p-value raster (index 6)
        with rasterio.open(raster_paths[6]) as src:
            rasters['pvalue'] = src.read(1)
            mask_array = (rasters['pvalue'] != nodata_value) & (rasters['pvalue'] != 255)
            rasters['pvalue'] = np.where(mask_array, rasters['pvalue'], np.nan)
    else:
        rasters['canopy_trend'] = np.full_like(rasters['biomass_mean'] , nodata_value)
        rasters['pvalue'] = np.full_like(rasters['biomass_mean'] , nodata_value)
        
    if raster_paths[7] is not None:        
        # Read deciduous fraction raster (index 7)
        with rasterio.open(raster_paths[7]) as src:
            rasters['deciduous'] = src.read(1)
            mask_array = (rasters['deciduous'] != nodata_value) & (rasters['deciduous'] != 255)
            rasters['deciduous'] = np.where(mask_array, rasters['deciduous'], np.nan)
    else:
        rasters['deciduous'] = np.full_like(rasters['biomass_mean'] , nodata_value)
        
    return rasters

def create_class_rasters(rasters, nodata_value=-9999):
    """
    Create classification rasters based on the input data.
    
    Parameters:
    -----------
    rasters : dict
        Dictionary with raster data
        
    Returns:
    --------
    dict
        Dictionary with added classification rasters
    """
    # Create age classes
    age_classes = ['non-forest','1-20', '21-36', '37-60', '61-80', '81-100', '101-150', '>150']
    age_bins = [0, 0, 20, 36, 60, 80, 100, 150, np.inf]

    rasters['age_class'] = np.full_like(rasters['age_mean'], 0, dtype=np.float32) #---------------

    for i, (lower, upper) in enumerate(zip(age_bins[:-1], age_bins[1:])):
        mask = (rasters['age_mean'] > lower) & (rasters['age_mean'] <= upper)
        rasters['age_class'][mask] = i
    print(f"\tFinished age class raster: {rasters['age_class'].shape}")

    if np.all(rasters['canopy_trend'] == nodata_value):
        print("\tNo canopy trends; no canopy trend class raster made.")
    else:
        # Create canopy trend classes
        # ['decline\n(strong)','decline\n(weak)','stable','increase\n(weak)','increase\n(strong)']
        rasters['trend_class'] = np.full_like(rasters['canopy_trend'], np.nan, dtype=np.float32) # prob want to put this before if statement so you get NaN values in the case of no trends.
        
        trend_mask_fill = (rasters['canopy_trend'] == 255)
        trend_mask_1 = rasters['canopy_trend'] < -2
        trend_mask_2 = (rasters['canopy_trend'] >= -2) & (rasters['canopy_trend'] < -0.5)
        trend_mask_4 = (rasters['canopy_trend'] <= 2) & (rasters['canopy_trend'] > 0.5)
        trend_mask_5 = rasters['canopy_trend'] > 2
        trend_mask_3 = ~(trend_mask_1 | trend_mask_2 | trend_mask_4 | trend_mask_5) & ~np.isnan(rasters['canopy_trend'])
        
        rasters['trend_class'][trend_mask_1] = 0  # Strong decline
        rasters['trend_class'][trend_mask_2] = 1  # Weak decline
        rasters['trend_class'][trend_mask_3] = 2  # Stable
        rasters['trend_class'][trend_mask_4] = 3  # Weak increase
        rasters['trend_class'][trend_mask_5] = 4  # Strong increase
        rasters['trend_class'][trend_mask_5] = 4  # Strong increase
        rasters['trend_class'][trend_mask_fill] = np.nan
        print(f"\tFinished trend class raster: {rasters['trend_class'].shape}")
        
        # Create p-value classes
        # ['not sig', 'sig (p<0.05)']
        rasters['pvalue_class'] = np.full_like(rasters['pvalue'], np.nan, dtype=np.float32)
        rasters['pvalue_class'][rasters['pvalue'] >= 0.05] = 0  # Not significant
        rasters['pvalue_class'][rasters['pvalue'] < 0.05] = 1   # Significant
        print(f"\tFinished pvalue class raster: {rasters['pvalue_class'].shape}")

    rasters['deciduous_class'] = np.full_like(rasters['deciduous'], np.nan, dtype=np.float32)
    if np.all(rasters['deciduous'] == nodata_value):
        print("\tNo deciduous fraction; all raster deciduous fraction class values set to NaN.")
    else:
        # Create deciduous fraction classes
        # ['conifer','mixed','deciduous']
        rasters['deciduous_class'][rasters['deciduous'] < 33] = 0  # Conifer
        rasters['deciduous_class'][(rasters['deciduous'] >= 33) & (rasters['deciduous'] <= 66)] = 1  # Mixed
        rasters['deciduous_class'][rasters['deciduous'] > 66] = 2  # Deciduous
        print(f"\tFinished deciduous class raster: {rasters['deciduous_class'].shape}")
    
    return rasters

def monte_carlo_carbon_accumulation(rasters, num_simulations=50, random_seed=None):
    """
    Perform Monte Carlo simulations to derive carbon accumulation estimates.
    
    Parameters:
    -----------
    rasters : dict
        Dictionary with raster data
    num_simulations : int
        Number of Monte Carlo simulations to run
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with carbon accumulation results
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initialize arrays for results
    rows, cols = rasters['shape']
    carbon_mean = np.full_like(rasters['biomass_mean'], np.nan)
    carbon_std = np.full_like(rasters['biomass_std'], np.nan)
    carbon_acc_mean = np.full_like(rasters['biomass_mean'], np.nan)
    carbon_acc_std = np.full_like(rasters['biomass_std'], np.nan)
    
    # Create arrays for simulation results
    sim_carbon = np.zeros((num_simulations, rows, cols))
    sim_age = np.zeros((num_simulations, rows, cols))
    sim_carbon_acc = np.zeros((num_simulations, rows, cols))
    
    # Get valid pixels (not NaN in both biomass and age)
    # Set nan to non-forest age mean value = 0.5, std = 0
    rasters['age_mean'] = np.nan_to_num(rasters['age_mean'], nan=0.0)  # < ---- se this to 0 or 0.5?
    rasters['age_std'] = np.nan_to_num(rasters['age_std'], nan=0)
    valid_mask = ~np.isnan(rasters['biomass_mean']) & ~np.isnan(rasters['age_mean'])
    valid_indices = np.where(valid_mask)

    # Ensure biomass standard deviation is non-negative
    rasters['biomass_std'] = np.maximum(rasters['biomass_std'], 0)
    
    # Ensure age standard deviation is non-negative
    rasters['age_std'] = np.maximum(rasters['age_std'], 0)
    
    # Perform Monte Carlo simulations
    for i in range(num_simulations):
        # Generate random biomass values
        sim_biomass = np.random.normal(
            rasters['biomass_mean'], 
            rasters['biomass_std']        )
        
        # Generate random age values
        sim_age[i] = np.random.normal(
            rasters['age_mean'], 
            rasters['age_std']
        )
        
        # Calculate carbon (50% of biomass)
        sim_carbon[i] = sim_biomass * 0.5
        
        # Calculate carbon accumulation (carbon / age)
        # Only for valid pixels with age > 0 to avoid division by zero
        valid_age_mask = (sim_age[i] > 0) & valid_mask
        sim_carbon_acc[i][valid_age_mask] = sim_carbon[i][valid_age_mask] / sim_age[i][valid_age_mask]
    
    # Calculate mean and standard deviation of carbon and carbon accumulation
    carbon_mean = np.mean(sim_carbon, axis=0)
    carbon_std = np.std(sim_carbon, axis=0, ddof=1)
    carbon_acc_mean = np.mean(sim_carbon_acc, axis=0)
    carbon_acc_std = np.std(sim_carbon_acc, axis=0, ddof=1)
    
    # Calculate confidence intervals for carbon accumulation
    t_value = stats.t.ppf(0.975, num_simulations - 1)
    carbon_acc_sem = carbon_acc_std / np.sqrt(num_simulations)
    carbon_acc_ci_lower = carbon_acc_mean - t_value * carbon_acc_sem
    carbon_acc_ci_upper = carbon_acc_mean + t_value * carbon_acc_sem
    
    # Calculate prediction intervals for carbon accumulation
    carbon_acc_pred_std = np.sqrt(carbon_acc_std**2 + carbon_acc_sem**2)
    carbon_acc_pi_lower = carbon_acc_mean - t_value * carbon_acc_pred_std
    carbon_acc_pi_upper = carbon_acc_mean + t_value * carbon_acc_pred_std
    
    print('\n--Completed C accumulation analysis.--\n')
    
    return {
        'carbon_mean': carbon_mean,
        'carbon_std': carbon_std,
        'carbon_acc_mean': carbon_acc_mean,
        'carbon_acc_std': carbon_acc_std,
        'carbon_acc_ci_lower': carbon_acc_ci_lower,
        'carbon_acc_ci_upper': carbon_acc_ci_upper,
        'carbon_acc_pi_lower': carbon_acc_pi_lower,
        'carbon_acc_pi_upper': carbon_acc_pi_upper,
        'sim_carbon': sim_carbon,
        'sim_carbon_acc': sim_carbon_acc
    }

import pandas as pd
from pandas.api.types import CategoricalDtype

def create_pixel_dataframe(rasters, carbon_results, 
                           vector_files=None, vector_columns=None, transform=None, crs=None):
    """
    Create a pandas DataFrame with pixel-level data with ordered categorical variables,
    and add attributes from vector files at each pixel location.
    
    Parameters:
    -----------
    rasters : dict
        Dictionary with raster data
    carbon_results : dict
        Dictionary with carbon accumulation results
    vector_files : dict, optional
        Dictionary with keys as identifiers and values as paths to geopackage files
    vector_columns : dict, optional
        Dictionary with keys matching vector_files and values as lists of column names to extract
    transform : affine.Affine, optional
        The transform of the raster data to convert pixel indices to coordinates
    crs : str or CRS object, optional
        The coordinate reference system of the raster data
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame with pixel-level data including vector attributes
    """
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    import geopandas as gpd
    from shapely.geometry import Point
    import rasterio
    from rasterio.transform import rowcol
    
    # Get valid pixels (not NaN in important variables)
    valid_mask = (
        ~np.isnan(rasters['biomass_mean']) & 
        ~np.isnan(rasters['age_mean']) &
        ~np.isnan(carbon_results['carbon_acc_mean'])
    )
    valid_indices = np.where(valid_mask)
    
    # Prepare data dictionary
    data = {
        'row': valid_indices[0],
        'col': valid_indices[1],
        'biomass': rasters['biomass_mean'][valid_indices],
        'biomass_std': rasters['biomass_std'][valid_indices],
        'age': rasters['age_mean'][valid_indices],
        'age_std': rasters['age_std'][valid_indices],
        'carbon': carbon_results['carbon_mean'][valid_indices],
        'carbon_std': carbon_results['carbon_std'][valid_indices],
        'carbon_acc': carbon_results['carbon_acc_mean'][valid_indices],
        'carbon_acc_std': carbon_results['carbon_acc_std'][valid_indices],
        'carbon_acc_ci_lower': carbon_results['carbon_acc_ci_lower'][valid_indices],
        'carbon_acc_ci_upper': carbon_results['carbon_acc_ci_upper'][valid_indices],
        'carbon_acc_pi_lower': carbon_results['carbon_acc_pi_lower'][valid_indices],
        'carbon_acc_pi_upper': carbon_results['carbon_acc_pi_upper'][valid_indices]
    }
    
    # Add classification data
    age_class_labels = ['non-forest','1-20', '21-36', '37-60', '61-80', '81-100', '101-150', '>150']
    age_cohort_labels = ['non-forest','re-growth forest','re-growth forest', 'young forest','young forest','young forest', 'mature forest', 'old-growth forest']
    trend_class_labels = ['decline\n(strong)', 'decline\n(weak)', 'stable', 'increase\n(weak)', 'increase\n(strong)']
    pvalue_class_labels = ['not sig', 'sig (p<0.05)']
    deciduous_class_labels = ['conifer', 'mixed', 'deciduous']
    
    # Add class values
    data['age_class_val'] = rasters['age_class'][valid_indices]
    data['trend_class_val'] = rasters['trend_class'][valid_indices]
    data['pvalue_class_val'] = rasters['pvalue_class'][valid_indices]
    data['deciduous_class_val'] = rasters['deciduous_class'][valid_indices]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add class labels using map
    df['age_class_raw'] = df['age_class_val'].map(lambda x: age_class_labels[int(x)] if not np.isnan(x) else np.nan)
    #df['age_cohort_raw'] = df['age_class_val'].map(lambda x: age_cohort_labels[int(x)] if not np.isnan(x) else np.nan)
    df['trend_class_raw'] = df['trend_class_val'].map(lambda x: trend_class_labels[int(x)] if not np.isnan(x) else 'no trend\navailable') #np.nan)
    df['pvalue_class_raw'] = df['pvalue_class_val'].map(lambda x: pvalue_class_labels[int(x)] if not np.isnan(x) else 'no trend\navailable') #np.nan)
    df['deciduous_class_raw'] = df['deciduous_class_val'].map(lambda x: deciduous_class_labels[int(x)] if not np.isnan(x) else np.nan)
    
    # Create ordered categorical types that preserve the order in the label lists
    age_cat_type = CategoricalDtype(categories=age_class_labels, ordered=True)
    #agecohort_cat_type = CategoricalDtype(categories=age_cohort_labels, ordered=True)
    trend_cat_type = CategoricalDtype(categories=trend_class_labels+['no trend\navailable'], ordered=True)
    pvalue_cat_type = CategoricalDtype(categories=pvalue_class_labels+['no trend\navailable'], ordered=True)
    deciduous_cat_type = CategoricalDtype(categories=deciduous_class_labels, ordered=True)
    
    # Convert to ordered categorical variables
    df['age_class'] = df['age_class_raw'].astype(age_cat_type)
    #df['age_cohort'] = df['age_cohort_raw'].astype(agecohort_cat_type)
    df['trend_class'] = df['trend_class_raw'].astype(trend_cat_type)
    df['pvalue_class'] = df['pvalue_class_raw'].astype(pvalue_cat_type)
    df['deciduous_class'] = df['deciduous_class_raw'].astype(deciduous_cat_type)
    
    # Drop the raw columns as they're no longer needed
    df = df.drop(['age_class_raw',
                  #'age_cohort_raw', 
                  'trend_class_raw', 'pvalue_class_raw', 'deciduous_class_raw'], axis=1)

    reclassification_map = dict(zip(age_class_labels, age_cohort_labels))
    df['age_cohort'] = df['age_class'].map(reclassification_map)

    # Process vector files if provided ------
    if vector_files and vector_columns and transform and crs:
        # Create a GeoDataFrame with pixel points
        # Convert pixel indices to coordinates
        df['x'], df['y'] = rasterio.transform.xy(transform, df['row'].values, df['col'].values)
        geometry = [Point(x, y) for x, y in zip(df['x'], df['y'])]
        gdf_pixels = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        
        # Process each vector file
        for file_id, file_path in vector_files.items():
            if file_id in vector_columns:
                # Read the vector file
                vector_data = gpd.read_file(file_path)
                
                # Make sure CRS match
                if vector_data.crs != gdf_pixels.crs:
                    vector_data = vector_data.to_crs(gdf_pixels.crs)
                
                # Keep only the columns of interest
                cols_to_keep = vector_columns[file_id]
                vector_data = vector_data[cols_to_keep + ['geometry']]
                
                # Perform spatial join
                # We use 'sjoin' with predicate='within' to find which polygon each point falls within
                joined = gpd.sjoin(gdf_pixels, vector_data, how='left', predicate='within')
                
                # Add prefixes to avoid column name conflicts
                prefix = f"{file_id}_"
                for col in cols_to_keep:
                    if col in joined.columns:
                        # Transfer the joined columns to the original dataframe
                        df[f"{prefix}{col}"] = joined[col]
        
        # Clean up coordinate columns if not needed
        if 'x' in df.columns and 'y' in df.columns and 'geometry' not in df.columns:
            df = df.drop(['x', 'y'], axis=1)
    
    return df

def calculate_area_and_total_carbon(df, pixel_area_ha, groupby_cols = ['age_class','age_cohort','trend_class', 'pvalue_class', 'deciduous_class']):
    """
    Calculate area and total carbon for each class combination (summarization of data frame).
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame with pixel-level data
    pixel_area_ha : float
        Area of a single pixel in hectares
        
    Returns:
    --------
    DataFrame
        Summary DataFrame with area and total carbon by class
    """
    # Group by all class combinations
    # Count pixels and calculate area
    summary = df.groupby(groupby_cols).size().reset_index(name='pixel_count')
    summary['area_ha'] = summary['pixel_count'] * pixel_area_ha
    
    # Calculate total carbon (biomass * 0.5 * area in ha) and convert to Pg (1 Pg = 10^9 Mg)
    carbon_stats = df.groupby(groupby_cols).agg({
        'biomass': ['mean', 'std', 'count'],
        'age': ['mean', 'std'],
        'carbon': ['mean', 'std', 'sum'],
        'carbon_acc': ['mean', 'std'],
        'carbon_acc_ci_lower': 'mean',
        'carbon_acc_ci_upper': 'mean',
        'carbon_acc_pi_lower': 'mean',
        'carbon_acc_pi_upper': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    carbon_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in carbon_stats.columns]
    
    # Merge with summary
    summary = pd.merge(summary, carbon_stats, on=groupby_cols)
    
    # Calculate total carbon in Pg (1 Pg = 10^9 Mg)
    #  note: this is a sum based on the mean biomass (density) of the class)
    summary['total_carbon_Pg'] = (summary['biomass_mean'] * 0.5 * summary['area_ha'] * 1e-9)
    
    # Preserve categorical order for plotting
    for col in groupby_cols:
        summary[col] = summary[col].astype(df[col].dtype)
    
    return summary
    
def calculate_nan_proportion(array):
    """
    Calculate the proportion of NaN pixels to total pixels in a numpy array.
    
    Parameters:
    array: numpy array (can be 1D, 2D, or multi-dimensional)
    
    Returns:
    float: proportion of NaN values (between 0 and 1)
    """
    # Count total number of elements
    total_pixels = array.size
    
    # Count NaN values
    nan_pixels = np.sum(np.isnan(array))
    
    # Calculate proportion
    nan_proportion = nan_pixels / total_pixels
    
    return nan_proportion
    
def calculate_monte_carlo_class_totals(rasters, carbon_results, class_combinations, pixel_area_ha, num_simulations=50):
    """
    Calculate Monte Carlo simulations for total carbon by class combination (summarization of rasters).
    
    Parameters:
    -----------
    rasters : dict
        Dictionary with raster data
    carbon_results : dict
        Dictionary with carbon accumulation results
    class_combinations : list
        List of unique class combinations
    pixel_area_ha : float
        Area of a single pixel in hectares
    num_simulations : int
        Number of Monte Carlo simulations
        
    Returns:
    --------
    DataFrame
        DataFrame with Monte Carlo simulation results for total carbon for each class combination
    """
    # Get simulation data
    sim_carbon = carbon_results['sim_carbon']
    
    # Initialize results list
    results = []
    
    # Extract class rasters for easier access
    age_class = rasters['age_class']
    trend_class = rasters['trend_class']
    pvalue_class = rasters['pvalue_class']
    deciduous_class = rasters['deciduous_class']
    
    # Define class labels
    age_class_labels = ['non-forest','1-20', '21-36', '37-60', '61-80', '81-100', '101-150', '>150']
    trend_class_labels = ['decline\n(strong)', 'decline\n(weak)', 'stable', 'increase\n(weak)', 'increase\n(strong)']+['no trend\navailable']
    pvalue_class_labels = ['not sig', 'sig (p<0.05)']+['no trend\navailable']
    deciduous_class_labels = ['conifer', 'mixed', 'deciduous']

    print(f"\tTotal class combos: {len(class_combinations)}")

    # Process each class combination
    n_class_combos_skipped = 0
    for age_idx, trend_idx, pval_idx, decid_idx in class_combinations:
        # Create mask for this class combination
        mask = (
            (age_class == age_idx) &
            (trend_class == trend_idx) &
            (pvalue_class == pval_idx) &
            (deciduous_class == decid_idx)
        )
        
        # Skip if no pixels in this class
        if not np.any(mask):
            #print(f'No pixels where age={age_class_labels[age_idx]}, trend={trend_class_labels[trend_idx]}, pval={pvalue_class_labels[pval_idx]}, decid={deciduous_class_labels[decid_idx]} ...')
            n_class_combos_skipped += 1
            continue
        
        # Calculate total carbon for each simulation
        total_carbon_sim = np.zeros(num_simulations)
        for i in range(num_simulations):
            # Carbon in Mg for this simulation (biomass * 0.5)
            carbon = sim_carbon[i] * mask
            # Total carbon in Pg (1 Pg = 10^9 Mg)
            #total_carbon_sim[i] = np.nansum(carbon) * 0.5 * pixel_area_ha * 1e-9 # <---- check here: why * 0.5 ?? Its already converted to C
            total_carbon_sim[i] = np.nansum(carbon) * pixel_area_ha * 1e-9
        
        # Calculate statistics
        mean_carbon = np.mean(total_carbon_sim)
        std_carbon = np.std(total_carbon_sim, ddof=1)
        
        # Calculate confidence interval
        t_value = stats.t.ppf(0.975, num_simulations - 1)
        sem = std_carbon / np.sqrt(num_simulations)
        ci_lower = mean_carbon - t_value * sem
        ci_upper = mean_carbon + t_value * sem
        
        # Calculate prediction interval
        pi_lower = mean_carbon - t_value * std_carbon
        pi_upper = mean_carbon + t_value * std_carbon
        
        # Store results
        results.append({
            'age_class': age_class_labels[age_idx],
            'trend_class': trend_class_labels[trend_idx],
            'pvalue_class': pvalue_class_labels[pval_idx],
            'deciduous_class': deciduous_class_labels[decid_idx],
            'total_carbon_Pg_mean': mean_carbon,
            'total_carbon_Pg_std': std_carbon,
            'total_carbon_Pg_ci_lower': ci_lower,
            'total_carbon_Pg_ci_upper': ci_upper,
            'total_carbon_Pg_pi_lower': pi_lower,
            'total_carbon_Pg_pi_upper': pi_upper
        })

    print(f"\Percent of class combos skipped: {100 * round(n_class_combos_skipped/len(class_combinations), 3)}")
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    print(f"\tCols: {result_df.columns}")
    
    # Create ordered categorical types
    age_cat_type = CategoricalDtype(categories=age_class_labels, ordered=True)
    trend_cat_type = CategoricalDtype(categories=trend_class_labels, ordered=True)
    pvalue_cat_type = CategoricalDtype(categories=pvalue_class_labels, ordered=True)
    deciduous_cat_type = CategoricalDtype(categories=deciduous_class_labels, ordered=True)
    
    # Convert to ordered categorical variables
    result_df['age_class'] = result_df['age_class'].astype(age_cat_type)
    result_df['trend_class'] = result_df['trend_class'].astype(trend_cat_type)
    result_df['pvalue_class'] = result_df['pvalue_class'].astype(pvalue_cat_type)
    result_df['deciduous_class'] = result_df['deciduous_class'].astype(deciduous_cat_type)
    
    return result_df

def plot_carbon_accumulation_by_age(df, output_dir):
    """
    Create boxplots of carbon accumulation by age class for different class combinations.
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame with pixel-level data
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    trend_colors = {
        'decline\n(strong)': '#a6611a',
        'decline\n(weak)': '#dfc27d',
        'stable': '#f5f5f5',
        'increase\n(weak)': '#80cdc1',
        'increase\n(strong)': '#018571',
        'no trend\navailable': 'black'
    }
    
    pvalue_colors = {
        'not sig': '#fc8d59',
        'sig (p<0.05)': '#91bfdb',
        'no trend\navailable': 'black'
    }
    
    deciduous_colors = {
        'conifer': '#5e3c99', 
        'mixed': '#ffffbf',
        'deciduous': '#fdb863'
    }
    
    # 1. Carbon accumulation by age class and trend class
    p1 = (
        ggplot(df, aes(x='age_class', y='carbon_acc', fill='trend_class')) +
        geom_boxplot(position=position_dodge(width=0.8), outlier_size=1) +
        scale_fill_manual(values=trend_colors) +
        scale_y_continuous(limits=(0,2)) +
        labs(
            x='Age Class (years)',
            y='Carbon Accumulation (Mg C/ha/year)',
            fill='Canopy Trend'
        ) +
        theme_minimal() +
        theme(
            axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(12, 8)
        ) +
        ggtitle('Carbon Accumulation by Age and Canopy Trend')
    )
    print(p1)
    p1.save(os.path.join(output_dir, 'carbon_acc_by_age_trend.png'), dpi=300, bg='white')
    
    # 2. Carbon accumulation by age class and significance class
    p2 = (
        ggplot(df, aes(x='age_class', y='carbon_acc', fill='pvalue_class')) +
        geom_boxplot(position=position_dodge(width=0.8), outlier_size=1) +
        scale_fill_manual(values=pvalue_colors) +
        scale_y_continuous(limits=(0,2)) +
        labs(
            x='Age Class (years)',
            y='Carbon Accumulation (Mg C/ha/year)',
            fill='Significance'
        ) +
        theme_minimal() +
        theme(
            axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(12, 8)
        ) +
        ggtitle('Carbon Accumulation by Age and Statistical Significance')
    )
    print(p2)
    p2.save(os.path.join(output_dir, 'carbon_acc_by_age_significance.png'), dpi=300, bg='white')
    
    # 3. Carbon accumulation by age class and deciduous class
    p3 = (
        ggplot(df, aes(x='age_class', y='carbon_acc', fill='deciduous_class')) +
        geom_boxplot(position=position_dodge(width=0.8), outlier_size=1) +
        scale_fill_manual(values=deciduous_colors) +
        scale_y_continuous(limits=(0,2)) +
        labs(
            x='Age Class (years)',
            y='Carbon Accumulation (Mg C/ha/year)',
            fill='Forest Type'
        ) +
        theme_minimal() +
        theme(
            axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(12, 8)
        ) +
        ggtitle('Carbon Accumulation by Age and Forest Type')
    )
    print(p3)
    p3.save(os.path.join(output_dir, 'carbon_acc_by_age_forest_type.png'), dpi=300, bg='white')
    
    # 4. Carbon accumulation confidence intervals by age and trend
    summary_df = df.groupby(['age_class', 'trend_class']).agg({
        'carbon_acc': 'mean',
        'carbon_acc_ci_lower': 'mean',
        'carbon_acc_ci_upper': 'mean',
        'carbon_acc_pi_lower': 'mean',
        'carbon_acc_pi_upper': 'mean'
    }).reset_index()
    
    p4 = (
        ggplot(summary_df, aes(x='age_class', y='carbon_acc', fill='trend_class')) +
        geom_point(aes(color='trend_class'), size=3, position=position_dodge(width=0.8)) +
        geom_errorbar(
            aes(ymin='carbon_acc_ci_lower', ymax='carbon_acc_ci_upper', color='trend_class'),
            width=0.2, position=position_dodge(width=0.8)
        ) +
        scale_fill_manual(values=trend_colors) +
        scale_color_manual(values={k: v for k, v in trend_colors.items()}) +
        scale_y_continuous(limits=(0,2)) +
        labs(
            x='Age Class (years)',
            y='Carbon Accumulation (Mg C/ha/year)',
            color='Canopy Trend',
            fill='Canopy Trend'
        ) +
        theme_minimal() +
        theme(
            axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(12, 8)
        ) +
        ggtitle('Carbon Accumulation by Age and Canopy Trend (with 95% Confidence Intervals)')
    )
    print(p4)
    p4.save(os.path.join(output_dir, 'carbon_acc_ci_by_age_trend.png'), dpi=300, bg='white')
    
    # 5. Carbon accumulation prediction intervals by age and forest type
    p5 = (
        ggplot(summary_df, aes(x='age_class', y='carbon_acc', fill='trend_class')) +
        geom_point(aes(color='trend_class'), size=3, position=position_dodge(width=0.8)) +
        geom_errorbar(
            aes(ymin='carbon_acc_pi_lower', ymax='carbon_acc_pi_upper', color='trend_class'),
            width=0.2, position=position_dodge(width=0.8)
        ) +
        scale_fill_manual(values=trend_colors) +
        scale_color_manual(values={k: v for k, v in trend_colors.items()}) +
        scale_y_continuous(limits=(0,2)) +
        labs(
            x='Age Class (years)',
            y='Carbon Accumulation (Mg C/ha/year)',
            color='Canopy Trend',
            fill='Canopy Trend'
        ) +
        theme_minimal() +
        theme(
            axis_text_x=element_text(angle=45, hjust=1),
            figure_size=(12, 8)
        ) +
        ggtitle('Carbon Accumulation by Age and Canopy Trend (with 95% Prediction Intervals)')
    )
    print(p5)
    p5.save(os.path.join(output_dir, 'carbon_acc_pi_by_age_trend.png'), dpi=300, bg='white')
    
    # 6. Plot facet grid by all class combinations
    # Select a subset to avoid too many plots
    subset_df = df.sample(min(100000, len(df)))
    
    p6 = (
        ggplot(subset_df, aes(x='age_class', y='carbon_acc', fill='deciduous_class')) +
        geom_boxplot(position=position_dodge(width=0.8), outlier_size=0.5) +
        scale_fill_manual(values=deciduous_colors) +
        facet_grid('trend_class ~ pvalue_class') +
        scale_y_continuous(limits=(0,2)) +
        labs(
            x='Age Class (years)',
            y='Carbon Accumulation (Mg C/ha/year)',
            fill='Forest Type'
        ) +
        theme_minimal() +
        theme(
            axis_text_x=element_text(angle=90, hjust=1, size=8),
            figure_size=(15, 12)
        ) +
        ggtitle('Carbon Accumulation by Age, Forest Type, Trend, and Significance')
    )
    print(p6)
    p6.save(os.path.join(output_dir, 'carbon_acc_facet_grid.png'), dpi=300, bg='white')


def CARBON_ACC_ANALYSIS(MAP_VERSION, TILE_NUM, num_simulations = 5, random_seed = None, N_PIX_SAMPLE = 100000, DO_WRITE_COG=False,
                               extent_type = 'tile', output_dir = "/projects/my-public-bucket/carbon_accumulation_analysis_TEST", local=False):

    if local:
        # Define output directory
        output_dir = os.path.join(output_dir, extent_type, f"{TILE_NUM:07}")    
    os.makedirs(output_dir, exist_ok=True)
    
    """
    Main function to execute the entire workflow.
    """
    
    BASIN_COG_DICT = {
        'topo_cog_fn':      get_cog_s3_path(TILE_NUM, mosaiclib.TOPO_TINDEX_FN_DICT['c2020updated_v2']),
        'landcover_cog_fn': get_cog_s3_path(TILE_NUM, mosaiclib.LC_TINDEX_FN_DICT['c2020updated']),
        'biomass_cog_fn':   get_cog_s3_path(TILE_NUM, mosaiclib.AGB_TINDEX_FN_DICT[MAP_VERSION]),#'2020_v2.1'
        'height_cog_fn':    get_cog_s3_path(TILE_NUM, mosaiclib.HT_TINDEX_FN_DICT[MAP_VERSION]), #'2020_v2.1'
        'standage_cog_fn':  get_cog_s3_path(TILE_NUM, mosaiclib.MISC_TINDEX_FN_DICT['FORESTAGE_BES_2020']),
        #'extent_gdf_fn':    TILE_GDF_FN,
        'tcc2020_cog_fn':   get_cog_s3_path(TILE_NUM, mosaiclib.MISC_TINDEX_FN_DICT['TCC_TP_2020']),
        'tccslope_cog_fn':  get_cog_s3_path(TILE_NUM, mosaiclib.MISC_TINDEX_FN_DICT['TCCTREND_TP_2020']),
        'tccpvalue_cog_fn': get_cog_s3_path(TILE_NUM, mosaiclib.MISC_TINDEX_FN_DICT['TCCTRENDPVAL_TP_2020']),
        'decidpred_cog_fn': get_cog_s3_path(TILE_NUM, mosaiclib.MISC_TINDEX_FN_DICT['DECPRED_AB_2015']),
    }

    # Define paths to your raster files
    file_paths = {
        'biomass': BASIN_COG_DICT['biomass_cog_fn'],
        'age': BASIN_COG_DICT['standage_cog_fn'],
        'landcover': BASIN_COG_DICT['landcover_cog_fn'],
        'topography': BASIN_COG_DICT['topo_cog_fn'],
        'canopy_cover': BASIN_COG_DICT['tcc2020_cog_fn'],
        'canopy_trend': BASIN_COG_DICT['tccslope_cog_fn'],
        'trend_pvalue': BASIN_COG_DICT['tccpvalue_cog_fn'],
        'deciduous_fraction': BASIN_COG_DICT['decidpred_cog_fn']
    }
    
    # Define class labels
    age_class_labels = ['non-forest','1-20', '21-36', '37-60', '61-80', '81-100', '101-150', '>150']
    age_cohort_labels = ['non-forest','re-growth forest','re-growth forest', 'young forest','young forest','young forest', 'mature forest', 'old-growth forest']
    trend_class_labels = ['decline\n(strong)', 'decline\n(weak)', 'stable', 'increase\n(weak)', 'increase\n(strong)']+['no trend\navailable']
    pvalue_class_labels = ['not sig', 'sig (p<0.05)']+['no trend\navailable']
    deciduous_class_labels = ['conifer', 'mixed', 'deciduous']

    # # Break if there is no TCC trend data
    # if file_paths['canopy_trend'] is None:
    #     print(f"Tile {TILE_NUM} doesnt have tree cover trends calc'd, skipping...")
    # else:
    
    # Define paths to raster files
    raster_paths = list(file_paths.values())
    
    # Define pixel area in hectares (30m resolution)
    # 30m x 30m = 900 sq. meters = 0.09 hectares
    pixel_area_ha = 0.09
    
    print("Reading and preparing raster data...")
    # Read and prepare raster data
    rasters = read_and_prepare_rasters(raster_paths)
    
    print("Creating classification rasters...")
    # Create classification rasters
    rasters = create_class_rasters(rasters)
    
    print("Performing Monte Carlo simulations for carbon accumulation...")
       
    carbon_results = monte_carlo_carbon_accumulation(rasters, num_simulations=num_simulations, random_seed=random_seed)

    if DO_WRITE_COG:
        carbon_results_stack = np.stack([carbon_results['carbon_acc_mean'], carbon_results['carbon_acc_std'] , 
                                 carbon_results['carbon_acc_ci_lower'], carbon_results['carbon_acc_ci_upper'], 
                                 carbon_results['carbon_acc_pi_lower'], carbon_results['carbon_acc_pi_upper']
                                ])
        carbon_results_stack_names = ["carbon_acc_mean", "carbon_acc_std", 
                              "carbon_acc_ci_lower", "carbon_acc_ci_upper", 
                              "carbon_acc_pi_lower", "carbon_acc_pi_upper"]
                
        # write COG to disk
        write_cog(
                    carbon_results_stack, 
                    os.path.join(output_dir, f'boreal_cacc_2020_{TILE_NUM:07}.tif'), 
                    rasters['crs'], 
                    rasters['transform'], 
                    carbon_results_stack_names, 
                    out_crs=rasters['crs'],
                    input_nodata_value= -9999
                     )

    # if np.count_nonzero(~np.isnan(rasters['canopy_trend'])) == 0:
    #     print(f"Tile {TILE_NUM} has all NaN tree cover trend data; wont proceed with C accumulation analysis.\nExiting.")
    #     return
    if file_paths['canopy_trend'] is None:
        print(f"Tile {TILE_NUM} has no tree cover trend data; wont proceed with C accumulation analysis.\nExiting.\n")
        return   

    print(f"Dictionary of class rasters: {rasters.keys()}\n")
    print(f"{rasters['trend_class'].min()},{rasters['trend_class'].max()}")
    print(f"Count of -9999: {np.sum(rasters['trend_class'] == -9999)}")
    print(f"Count of nan: {np.sum(np.isnan(rasters['trend_class']))}")
    print(f"Count of unique: {np.unique(rasters['trend_class'], return_counts=True)}")
    
    if np.all(np.isnan(rasters['trend_class'] )):
        #  Need to catch case where trend raster is all NaN due to a tile available that is fully outside of boreal and has all NaN value
        print('\tNo canopy trends; no canopy trend class raster made.\nExiting.\n')
        return

    print("Creating pixel-level dataframe...")
    pixel_df = create_pixel_dataframe(
        rasters=rasters, 
        carbon_results=carbon_results,
        vector_files={
            'ecoregions': 'https://maap-ops-workspace.s3.amazonaws.com/shared/montesano/databank/wwf_terr_ecos.gpkg', 
            #'boreal': 'https://maap-ops-workspace.s3.amazonaws.com/shared/montesano/databank/arc/wwf_circumboreal_Dissolve.gpkg'
        },
        vector_columns={
            'ecoregions': ['ECO_NAME', 'REALM'],
            #'boreal': ['REALM']
        },
        transform=rasters['transform'],  # Get this from your open raster file
        crs=rasters['crs']  # Get this from your open raster file
    )
    pixel_df['ID'] = TILE_NUM

    if N_PIX_SAMPLE is None:
        print(f"Setting sample size to full length of data frame ({len(pixel_df)})...")
        sample_size = len(pixel_df)
    else:
        print(f"Setting sample size to a specific number of rows from the data frame ({N_PIX_SAMPLE})...")
        # Save pixel dataframe (sample to avoid very large files)
        sample_size = min(N_PIX_SAMPLE, len(pixel_df))
        
    pixel_df.sample(sample_size).to_csv(os.path.join(output_dir, f"pixel_data_sample_{TILE_NUM:07}.csv"), index=False)
    print(f"Saved sample of {sample_size} pixels to pixel_data_sample_{TILE_NUM:07}.csv")

    # Sample 10% of the rows
    SAMP_FRAC=0.1
    sampled_df = pixel_df.sample(frac=SAMP_FRAC, random_state=random_seed)
    sampled_df.to_csv(os.path.join(output_dir, f"pixel_data_sample10pct_{TILE_NUM:07}.csv"), index=False)
    print(f"Additionally, saved sample of {SAMP_FRAC*100}% of pixels to pixel_data_sample{int(SAMP_FRAC*100)}pct_{TILE_NUM:07}.csv")
    
    print("Calculating area and total carbon by class...")
    # Calculate area and total carbon by class
    summary_df = calculate_area_and_total_carbon(pixel_df, pixel_area_ha, 
                                                groupby_cols = ['ecoregions_REALM','ecoregions_ECO_NAME', 'age_class','age_cohort',
                                                                'trend_class', 'pvalue_class', 'deciduous_class']) 
    summary_df = summary_df[summary_df['pixel_count']>0]
    summary_df.to_csv(os.path.join(output_dir, f"summary_by_class_{TILE_NUM:07}.csv"), index=False)
    
    print("Calculating Monte Carlo totals for carbon by class...")
    # Get unique class combinations
    class_combinations = []
    for age_idx, _ in enumerate(age_class_labels):  # 8 age classes
        for trend_idx, _ in enumerate(trend_class_labels):  # 5 trend classes
            for pval_idx, _ in enumerate(pvalue_class_labels):  # 2 p-value classes
                for decid_idx, _ in enumerate(deciduous_class_labels):  # 3 deciduous classes
                    class_combinations.append((age_idx, trend_idx, pval_idx, decid_idx))
                    
    if np.all(rasters['canopy_trend'] == 255):
        #  Need to catch case where trend raster is all NaN due to a tile available that is fully outside of boreal and has all NaN value
        print('\tNo canopy trends; no canopy trend class raster made.')
        return
    
    # Calculate Monte Carlo totals for carbon by class
    monte_carlo_totals = calculate_monte_carlo_class_totals(
        rasters, carbon_results, class_combinations, pixel_area_ha, num_simulations
    )
    monte_carlo_totals.to_csv(os.path.join(output_dir, f"monte_carlo_totals_{TILE_NUM:07}.csv"), index=False)
    
    if False:
        print("Creating visualizations...")
        # Create visualizations
        plot_carbon_accumulation_by_age(pixel_df, output_dir)
    
    # Calculate grand total carbon with confidence intervals
    grand_total_carbon = monte_carlo_totals['total_carbon_Pg_mean'].sum()
    grand_total_ci_lower = monte_carlo_totals['total_carbon_Pg_ci_lower'].sum()
    grand_total_ci_upper = monte_carlo_totals['total_carbon_Pg_ci_upper'].sum()
    grand_total_pi_lower = monte_carlo_totals['total_carbon_Pg_pi_lower'].sum()
    grand_total_pi_upper = monte_carlo_totals['total_carbon_Pg_pi_upper'].sum()
    
    print("\n===== CARBON STOCK RESULTS =====")
    print(f"Total Carbon Stock: {grand_total_carbon:.9f} Pg C ({grand_total_carbon/1e-9:.4f} Mg C)")
    print(f"95% Confidence Interval: ({grand_total_ci_lower:.9f}, {grand_total_ci_upper:.9f}) Pg C")
    print(f"95% Prediction Interval: ({grand_total_pi_lower:.9f}, {grand_total_pi_upper:.9f}) Pg C")
    
    # Generate summary statistics by different groupings
    print("\nGenerating summary statistics by various groupings...")
    
    # By age class
    age_summary = monte_carlo_totals.groupby('age_class').agg({
        'total_carbon_Pg_mean': 'sum',
        'total_carbon_Pg_ci_lower': 'sum',
        'total_carbon_Pg_ci_upper': 'sum'
    }).reset_index()
    # Preserve categorical order
    age_summary['age_class'] = age_summary['age_class'].astype(
        CategoricalDtype(categories=age_class_labels, ordered=True))
    age_summary.to_csv(os.path.join(output_dir, f"age_class_summary_{TILE_NUM:07}.csv"), index=False)
    
    # By trend class
    trend_summary = monte_carlo_totals.groupby('trend_class').agg({
        'total_carbon_Pg_mean': 'sum',
        'total_carbon_Pg_ci_lower': 'sum',
        'total_carbon_Pg_ci_upper': 'sum'
    }).reset_index()
    # Preserve categorical order
    trend_summary['trend_class'] = trend_summary['trend_class'].astype(
        CategoricalDtype(categories=trend_class_labels, ordered=True))
    trend_summary.to_csv(os.path.join(output_dir, f"trend_class_summary_{TILE_NUM:07}.csv"), index=False)
    
    # By deciduous class
    deciduous_summary = monte_carlo_totals.groupby('deciduous_class').agg({
        'total_carbon_Pg_mean': 'sum',
        'total_carbon_Pg_ci_lower': 'sum',
        'total_carbon_Pg_ci_upper': 'sum'
    }).reset_index()
    # Preserve categorical order
    deciduous_summary['deciduous_class'] = deciduous_summary['deciduous_class'].astype(
        CategoricalDtype(categories=deciduous_class_labels, ordered=True))
    deciduous_summary.to_csv(os.path.join(output_dir, f"deciduous_class_summary_{TILE_NUM:07}.csv"), index=False)
    
    # Write to a JSON file
    with open(os.path.join(output_dir, 'input_data.json'), 'w') as json_file:
        json.dump(BASIN_COG_DICT, json_file)
        
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_version", type=str, default='2020_v3.0', help="String indicating the tindex filename of the AGB and HT data in their dicts")
    parser.add_argument("--in_tile_num", type=int, help="The id of a tile that will define the bounds of the raster stacking")
    parser.add_argument("--output_dir", type=str, help="The path for the JSON files to be written")
    parser.add_argument("--n_sims", type=int, default=5, help="The number of monte carlo simulations to sample from AGB and age data (w/ mean, stdev)")
    parser.add_argument("--seed", type=int, default=123, help="The random seed to set for reproducability")
    parser.add_argument("--n_pix_samples", type=int, default=None, help="The number of pixels to sample from the pixel-level dataframe")
    parser.add_argument("--extent_type", type=str, choices=['tile','hydrobasin'], default='tile', help="The type of extent that is used w/ in_tile_num to specify output sub-dir.")
    parser.add_argument('--do_write_cog', dest='do_write_cog', action='store_true', help='Write a cloud-optimized geotiff of results.')
    parser.set_defaults(do_write_cog=False)
    args = parser.parse_args()    
    
    '''
    Run the carbon accumulation analysis
    '''

    CARBON_ACC_ANALYSIS(args.map_version, args.in_tile_num, num_simulations=args.n_sims, random_seed=args.seed, N_PIX_SAMPLE=args.n_pix_samples, 
                        DO_WRITE_COG=args.do_write_cog, extent_type=args.extent_type, output_dir=args.output_dir)

if __name__ == "__main__":
    main()