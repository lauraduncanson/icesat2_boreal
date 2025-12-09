import numpy as np

def get_default_age_classification():
    """Return default age cohort classification scheme."""
    return {
        'non_forest': [0, 0],
        'young_forest': [1, 20],
        'maturing_forest': [21, 80],
        'mature_forest': [81, 200],
        'old_growth_forest': [201, 9999]  # Using 9999 instead of inf for JSON compatibility
    }

def get_kendall_trend_class_labels():
    """
    Return the Kendall Tau trend class labels from compute_trends.py.
    Maps the class values (0-10) to descriptive labels.
    """
    return {
        0: 'No data',
        1: 'Strong sig. positive',
        2: 'Moderate sig. positive',
        3: 'Weak sig. positive',
        4: 'Very weak sig. positive',
        5: 'Non-sig. positive',
        6: 'Non-sig. negative',
        7: 'Very weak sig. negative',
        8: 'Weak sig. negative',
        9: 'Moderate sig. negative',
        10: 'Strong sig. negative'
    }

def get_trend_category_mapping():
    """
    Return mapping from Kendall Tau classes to simpler trend categories.
    Groups the detailed Kendall classes into 5 simplified categories.
    """
    return {
        0: -1,   # No data -> ignore
        1: 4,    # Strong sig. positive -> strong increase
        2: 4,    # Moderate sig. positive -> strong increase
        3: 3,    # Weak sig. positive -> moderate increase
        4: 3,    # Very weak sig. positive -> moderate increase
        5: 2,    # Non-sig. positive -> stable
        6: 2,    # Non-sig. negative -> stable
        7: 1,    # Very weak sig. negative -> moderate decline
        8: 1,    # Weak sig. negative -> moderate decline
        9: 0,    # Moderate sig. negative -> strong decline
        10: 0    # Strong sig. negative -> strong decline
    }

def classify_age_cohorts(age_array, classification_scheme):
    """
    Classify age values into cohorts using provided scheme.
    
    Parameters:
    age_array: numpy array of age values
    classification_scheme: dict with age ranges
    """
    # Initialize cohort array
    cohorts = np.full_like(age_array, -1, dtype=np.int8)  # -1 for unclassified
    
    for i, (cohort_name, age_range) in enumerate(classification_scheme.items()):
        min_age, max_age = age_range
        
        if max_age >= 9999:  # Handle large numbers as infinity
            mask = age_array > min_age
        else:
            mask = (age_array >= min_age) & (age_array <= max_age)
        
        valid_mask = ~np.isnan(age_array)
        cohorts[mask & valid_mask] = i
    
    return cohorts, list(classification_scheme.keys())

def remap_trend_classes(kendall_classes):
    """
    Remap detailed Kendall Tau classes to simplified trend categories.
    
    Parameters:
    kendall_classes: numpy array of Kendall Tau class values (0-10)
    
    Returns:
    Numpy array with remapped trend categories (0-4)
    """
    # Get the mapping dictionary
    mapping = get_trend_category_mapping()
    
    # Create output array
    trend_categories = np.full_like(kendall_classes, -1, dtype=np.int8)
    
    # Apply mapping
    for kendall_class, trend_category in mapping.items():
        trend_categories[kendall_classes == kendall_class] = trend_category
    
    return trend_categories

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
