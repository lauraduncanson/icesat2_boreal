import pandas as pd
import geopandas as gpd
import rasterio as rio
import os
import numpy as np

import sys
sys.path.append('/projects/code/icesat2_boreal/notebooks/3.Gridded_product_development')
from CovariateUtils import *

import itertools
import copy

#from notebooks.general.covariateutils import get_index_tile
#import notebooks.general.CovariateUtils 
#from CovariateUtils import get_index_tile

# import the MAAP package
from maap.maap import MAAP

# create MAAP class
maap = MAAP()

def get_h5_list(tile_num, tile_fn="/projects/maap-users/alexdevseed/boreal_tiles.gpkg", layer="boreal_tiles_albers",DATE_START='06-01', DATE_END='09-30', YEARS=[2019, 2020, 2021]):
    '''
    Return a list of ATL08 h5 names that intersect a tile for a give date range across a set of years
    '''
    tile_id = CovariateUtils.get_index_tile(tile_fn, tile_num, buffer=0, layer = layer)

    in_bbox = ",".join(str(coord) for coord in tile_id['bbox_4326'])
    
    print("\tTILE_NUM: {} ({})".format(tile_num, in_bbox) )
    
    out_crs = tile_id['tile_crs']
    
    DATE_START = DATE_START + 'T00:00:00Z' # SUMMER start
    DATE_END = DATE_END + 'T23:59:59Z' # SUMMER end
    
    date_filters = [f'{year}-{DATE_START},{year}-{DATE_END}' for year in YEARS]
    
    base_query = {
    'short_name':"ATL08",
    'version':"003",
    'bounding_box':in_bbox
    }

    #q3 = [build_query(copy.copy(base_query), date_filter) for date_filter in date_filters]
    queries = [dict(base_query, temporal=date_filter) for date_filter in date_filters]
    
    # query CMR as many seasons as necessary
    result_chain = itertools.chain.from_iterable([maap.searchGranule(**query) for query in queries])
    
    # This is the list of ATL08 that intersect the tile bounds
    # Use this list of ATL08 to identify the ATL08 h5/CSV files you have already DPS'd.
    # get the s3 urls for granules we want to process
    granules = [item.getDownloadUrl() for item in result_chain]
    
    # Convert to just the h5 basenames (removing the s3 url)
    out_file_list = [os.path.basename(x) for x in granules]
    
    print("\t\t# ATL08 for tile {}: {}".format(tile_num, len(out_file_list)) )
    
    return(out_h5_list)

def extract_value_gdf(r_fn, pt_gdf, bandnames: list, reproject=True, TEST=False):
    """Extract raster band values to the obs of a geodataframe
    """

    print("\tOpen the raster and store metadata...")
    r_src = rio.open(r_fn)
    
    if reproject:
        print("\tRe-project points to match raster...")
        pt_gdf = pt_gdf.to_crs(r_src.crs)
    
    for idx, bandname in enumerate(bandnames):
        bandnum = idx + 1
        if TEST: print("Read as a numpy masked array...")
        r = r_src.read(bandnum, masked=True)
        
        if TEST: print(r.dtype)

        pt_coord = [(pt.x, pt.y) for pt in pt_gdf.geometry]

        # Use 'sample' from rasterio
        if TEST: print("Create a generator for sampling raster...")
        pt_sample = r_src.sample(pt_coord, bandnum)
        
        if TEST:
            for i, val in enumerate(r_src.sample(pt_coord, bandnum)):
                print("point {} value: {}".format(i, val))
            
        if TEST: print("Use generator to evaluate (sample)...")
        pt_sample_eval = np.fromiter(pt_sample, dtype=r.dtype)

        if TEST: print("Deal with no data...")
        pt_sample_eval_ma = np.ma.masked_equal(pt_sample_eval, r_src.nodata)
        #pt_gdf[bandname] = pd.Categorical(pt_sample_eval_ma.astype(int).filled(-1))
        pt_gdf[bandname] = pt_sample_eval_ma.astype(float).filled(np.nan)
        
        print('\tDataframe has new raster value column: {}'.format(bandname))
        r = None
        
    r_src.close()
    
    print('Returning re-projected points with {} new raster value column: {}'.format(len(bandnames), bandnames))
    return(pt_gdf)

def get_covar_fn_list(rootDir, tile_num):
    '''
    Get a list of covar filenames using a root dir and a tile_num string that is found in each covar file name
    '''
    covar_tile_list = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if fname.endswith('.tif') and str(tile_num) in fname:
                covar_tile_list.append(os.path.join(dirName , fname))
                
    return(covar_tile_list)