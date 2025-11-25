import numpy as np
from scipy import ndimage
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

def read_raster(raster_path, band=1):
    """
    Read raster data from file path
    
    Parameters:
    -----------
    raster_path : str
        Path to raster file
    band : int
        Band number to read (1-indexed)
    
    Returns:
    --------
    numpy.ndarray
        Raster data for the specified band
    """
    with rasterio.open(raster_path) as src:
        data = src.read(band)
    
    return data

def read_raster_with_profile(raster_path, band=1):
    """
    Read raster data with geospatial profile information
    
    Parameters:
    -----------
    raster_path : str
        Path to raster file
    band : int
        Band number to read (1-indexed)
    
    Returns:
    --------
    tuple
        (data, transform, crs) - raster data, affine transform, and CRS
    """
    with rasterio.open(raster_path) as src:
        data = src.read(band)
        transform = src.transform
        crs = src.crs
    
    return data, transform, crs

def get_latitude_grid(shape, transform, crs):
    """
    Create latitude grid for the raster using rasterio transform
    
    Parameters:
    -----------
    shape : tuple
        (height, width) of the raster
    transform : rasterio.Affine
        Affine transform of the raster
    crs : rasterio.crs.CRS
        Coordinate reference system of the raster
    
    Returns:
    --------
    numpy.ndarray
        2D array of latitude values for each pixel
    """
    height, width = shape
    
    # Create coordinate grids for all pixels
    rows, cols = np.mgrid[0:height, 0:width]
    
    # Convert pixel coordinates to projected coordinates
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    xs = np.array(xs).reshape(shape)
    ys = np.array(ys).reshape(shape)
    
    # Set up transformer from the custom Albers to WGS84 (lat/lon)
    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    
    # Transform projected coordinates to lat/lon
    # Note: transformer expects (x, y) but returns (lon, lat)
    lons, lats = transformer.transform(xs.ravel(), ys.ravel())
    
    # Reshape back to original grid shape
    latitude_grid = np.array(lats).reshape(shape)
    
    return latitude_grid

def pad_array_to_shape(array, target_shape):
    """
    Pad array to target shape by replicating last row/column as needed.
    
    Parameters:
    -----------
    array : numpy array
        Input array to pad
    target_shape : tuple
        Target shape (height, width)
        
    Returns:
    --------
    padded_array : numpy array
        Array padded to target shape
    """
    import numpy as np
    
    current_shape = array.shape
    target_height, target_width = target_shape
    current_height, current_width = current_shape
    
    # Start with the original array
    padded_array = array.copy()
    
    # Pad rows if needed (replicate last row)
    if current_height < target_height:
        rows_to_add = target_height - current_height
        last_row = padded_array[-1:, :]  # Get last row, keep 2D
        repeated_rows = np.repeat(last_row, rows_to_add, axis=0)
        padded_array = np.vstack([padded_array, repeated_rows])
    
    # Pad columns if needed (replicate last column)
    if current_width < target_width:
        cols_to_add = target_width - current_width
        last_col = padded_array[:, -1:]  # Get last column, keep 2D
        repeated_cols = np.repeat(last_col, cols_to_add, axis=1)
        padded_array = np.hstack([padded_array, repeated_cols])
    
    return padded_array

def create_comprehensive_mask(agb_stack, landcover, slope, latitude,
                            biomass_threshold=10,
                            trend_threshold_lo=3,
                            trend_threshold_hi=8,
                            high_latitude_threshold=60,
                            slope_threshold=5,
                            window_size=11):
    """
    Create comprehensive mask combining multiple criteria
    
    Parameters:
    -----------
    agb_stack : numpy.ndarray
        Stack of AGB rasters (time, height, width)
    landcover : numpy.ndarray
        Land cover raster
    slope : numpy.ndarray
        Slope raster
    latitude : numpy.ndarray
        Latitude grid
    biomass_threshold : float
        Biomass threshold for low biomass identification
    trend_threshold_low : float
        Kendall tau threshold for moderate declining trends (default 3)
    trend_threshold_high : float
        Kendall tau threshold for strong declining trends (default 8)
    high_latitude_threshold : float
        Latitude threshold (degrees N)
    slope_threshold : float
        Slope threshold (degrees) for moss/lichen masking below 60N
    window_size : int
        Window size for spatial context filtering
    
    Returns:
    --------
    numpy.ndarray
        Boolean mask (True = keep pixel, False = mask out)
    """
    
    # Mask 1: Multi-criteria filtering for moss/lichen pixels
    print(f'Shapes or arrays:\n\tagb stack: {agb_stack.shape}\n\tlandcover: {landcover.shape}\n\tslope: {slope.shape}')
    multi_criteria_moss_lichen_mask = do_multi_criteria_moss_lichen_mask(
        agb_stack, landcover, slope, latitude, 
        biomass_threshold, high_latitude_threshold, slope_threshold
    )

    # These are experimental and not tested/implemented
    #
    # # Mask 2: Statistical outlier detection (now with trend thresholds)
    # outlier_mask = detect_biomass_outliers(
    #     agb_stack, biomass_threshold, 
    #     trend_threshold_lo, trend_threshold_hi
    # )
    
    # # Mask 3: Spatial context filtering
    # spatial_mask = spatial_context_mask(
    #     agb_stack, landcover, latitude, high_latitude_threshold, 
    #     slope, slope_threshold, window_size
    # )
    
    # Combine all masks - keep pixel if ANY method says to keep it
    # This is conservative - only mask pixels that ALL methods agree should be masked
    #final_mask = multi_criteria_moss_lichen_mask & outlier_mask & spatial_mask
    final_mask = multi_criteria_moss_lichen_mask
    
    return final_mask

def do_multi_criteria_moss_lichen_mask(agb_stack, landcover, slope, latitude,
                             biomass_threshold, high_latitude_threshold, slope_threshold):
    """
    Method 2: Multi-criteria filtering with latitude-dependent moss/lichen handling
        use to mask out ESA Worldcover v1 2020 moss/lichen (value=100) extents that are:
            1. good in the high latitudes (Seward Pen., Brooks Range)
            2. not good in southwestern Canada, where this class includes recent harvest where recovery of woody biomass could be ongoing.

        This mask attempts to overcome the problem of applying this moss/lichen universally with criteria based on latitude, slope, and biomass that:
            1. masks out the true moss/lichen (likely no woody biomass recovery in progress)
            2. retains moss/lichen pixels often erroneously classifed over flat (slope < slope_threshold) and low biomass in low latitudes
    """
    import numpy as np
    
    # Get reference shape from agb_stack (assuming it's 3D: time, height, width)
    if agb_stack.ndim == 3:
        ref_shape = agb_stack.shape[1:]  # Get height, width from (time, height, width)
    else:
        ref_shape = agb_stack.shape  # Assume 2D
    
    # Check and pad arrays to match reference shape if needed
    if landcover.shape != ref_shape:
        print(f"Padding landcover from {landcover.shape} to {ref_shape}")
        landcover = pad_array_to_shape(landcover, ref_shape)
    
    if slope.shape != ref_shape:
        print(f"Padding slope from {slope.shape} to {ref_shape}")
        slope = pad_array_to_shape(slope, ref_shape)
    
    if latitude.shape != ref_shape:
        print(f"Padding latitude from {latitude.shape} to {ref_shape}")
        latitude = pad_array_to_shape(latitude, ref_shape)
    
    # Low biomass identification
    mean_biomass = np.mean(agb_stack, axis=0)
    low_biomass = mean_biomass < biomass_threshold
    
    # Latitude-dependent moss/lichen masking
    high_latitude = latitude >= high_latitude_threshold
    low_latitude = latitude < high_latitude_threshold
    
    # Above 60N: mask all moss/lichen (100) in low biomass areas
    moss_lichen_mask_high_lat = (landcover == 100) & high_latitude & low_biomass
    
    # Below 60N: only mask moss/lichen on steep slopes (> slope_threshold)
    steep_slopes = slope > slope_threshold
    moss_lichen_mask_low_lat = (landcover == 100) & low_biomass & low_latitude & steep_slopes
    
    # Combine moss/lichen masks
    problematic_moss_lichen = moss_lichen_mask_high_lat | moss_lichen_mask_low_lat
    
    # Return keep mask (invert the problematic areas)
    return ~problematic_moss_lichen

def detect_biomass_outliers(agb_stack, biomass_threshold, 
                           trend_threshold_low=3, trend_threshold_high=8):
    """
    Method 3: Detect pixels with suspicious biomass patterns and trends
    
    Parameters:
    -----------
    agb_stack : numpy.ndarray
        Stack of AGB rasters (time, height, width)
    biomass_threshold : float
        Biomass threshold for low biomass identification
    trend_threshold_low : float
        Kendall tau threshold for moderate declining trends
    trend_threshold_high : float
        Kendall tau threshold for strong declining trends
    
    Returns:
    --------
    numpy.ndarray
        Boolean mask (True = keep pixel, False = mask out)
    """
    mean_biomass = np.mean(agb_stack, axis=0)
    std_biomass = np.std(agb_stack, axis=0)
    
    # Flag pixels with very low biomass but high variability
    # (might indicate noisy/unreliable estimates)
    low_biomass = mean_biomass < biomass_threshold
    
    # Avoid division by zero and handle very small values
    cv = np.divide(std_biomass, mean_biomass, out=np.zeros_like(std_biomass), where=mean_biomass!=0)
    high_variability = cv > 0.5
    
    # Compute simple trend indicator (could be replaced with actual Kendall tau if available)
    # This is a placeholder - you might want to compute actual trends here
    time_steps = np.arange(agb_stack.shape[0])
    trend_slopes = np.zeros(agb_stack.shape[1:])
    
    for i in range(agb_stack.shape[1]):
        for j in range(agb_stack.shape[2]):
            if not np.any(np.isnan(agb_stack[:, i, j])):
                # Simple linear trend slope as proxy for Kendall tau class
                slope, _ = np.polyfit(time_steps, agb_stack[:, i, j], 1)
                # Convert slope to approximate trend class (this is a rough approximation)
                # You might want to replace this with actual Kendall tau computation
                if slope < -2:  # Strong decline
                    trend_slopes[i, j] = 9  # Strong decline class
                elif slope < -1:  # Moderate decline
                    trend_slopes[i, j] = 5  # Moderate decline class
                else:
                    trend_slopes[i, j] = 1  # Stable/increasing
    
    # Flag pixels with suspicious combinations:
    # 1. High variability in low biomass areas
    suspicious_variability = low_biomass & high_variability
    
    # 2. Strong declining trends in low biomass areas (likely spurious)
    suspicious_strong_trends = low_biomass & (trend_slopes >= trend_threshold_high)
    
    # 3. Even moderate declining trends in very low biomass areas might be suspicious
    very_low_biomass = mean_biomass < (biomass_threshold * 0.5)  # Half the threshold
    suspicious_moderate_trends = very_low_biomass & (trend_slopes >= trend_threshold_low)
    
    # Combine all suspicious patterns
    suspicious_pixels = (suspicious_variability | 
                        suspicious_strong_trends | 
                        suspicious_moderate_trends)
    
    return ~suspicious_pixels

def spatial_context_mask(agb_stack, landcover, latitude, high_latitude_threshold,
                        slope, slope_threshold, window_size):
    """
    Method 4: Spatial context filtering with latitude considerations
    """
    mean_agb = np.mean(agb_stack, axis=0)
    
    # Calculate local neighborhood mean
    neighborhood_mean = ndimage.uniform_filter(mean_agb, size=window_size)
    
    # Calculate neighborhood land cover mode (most common value)
    def mode_filter(x):
        """Calculate mode, handling edge cases"""
        x_int = x.astype(int)
        x_int = x_int[~np.isnan(x_int)]  # Remove NaN values
        if len(x_int) == 0:
            return 0
        return np.bincount(x_int).argmax()
    
    neighborhood_lc_mode = ndimage.generic_filter(
        landcover.astype(float), 
        mode_filter, 
        size=window_size
    )
    
    # Different logic for high vs low latitude
    high_latitude = latitude >= high_latitude_threshold
    low_latitude = latitude < high_latitude_threshold
    
    # High latitude: mask low biomass pixels in predominantly moss/lichen neighborhoods
    high_lat_context = (
        (mean_agb < 10) & 
        (neighborhood_mean < 15) & 
        (neighborhood_lc_mode == 100) & 
        high_latitude
    )
    
    # Low latitude: only mask if also on steep slopes and in moss/lichen context
    steep_slopes = slope > slope_threshold
    low_lat_context = (
        (mean_agb < 10) & 
        (neighborhood_mean < 15) & 
        (neighborhood_lc_mode == 100) & 
        steep_slopes & 
        low_latitude
    )
    
    # Combine contexts
    mask_out = high_lat_context | low_lat_context
    
    return ~mask_out