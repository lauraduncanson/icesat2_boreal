import pandas as pd
import geopandas
import rasterio as rio
from rasterio.crs import CRS
import os
import numpy as np

import sys
#sys.path.append('/projects/code/icesat2_boreal/notebooks/3.Gridded_product_development')
#from CovariateUtils import *
#import CovariateUtils

import itertools
import copy

#from notebooks.general.covariateutils import get_index_tile
#import notebooks.general.CovariateUtils 
#from CovariateUtils import get_index_tile

# import the MAAP package
from maap.maap import MAAP

# create MAAP class
maap = MAAP(maap_host='api.ops.maap-project.org')

def get_index_tile(vector_path: str, tile_id: int, buffer: float = 0, layer: str = None):
    '''
    Given a vector tile index, select by id the polygon and return
    GPKG is the recommended vector format - single file, includes projection, can contain multiple variants and additional information.
    TODO: should it be a class or dict
    
    
    vector_path: str
        Path to GPKG file
    buffer: float
        Distance to buffer geometry in units of layer
    tile_id: int
        Tile ID to extract/build info for
        
    returns:
        geopandas.geodataframe.GeoDataFrame,
            Polygon in original crs
        geopandas.geoseries.GeoSeries,
            Polygon of Buffered in original crs
        list,
            Bounds of original polygon
        rasterio.crs.CRS,
            Coordinate Reference System of original tile
        geopandas.geodataframe.GeoDataFrame,
            4326 Polygon
        list,
            Bounds in 4326
        geopandas.geoseries.GeoSeries,
            Polygon of Buffered in 4326
        list
            Buffered Bounds in 4326
    Usage:
    get_index_tile(
        vector_path = '/projects/maap-users/alexdevseed/boreal_tiles.gpkg',
        tile_id = 30542,
        buffer = 120
        )
    
    '''
    
    tile_parts = {}

    if layer is None:
        layer = os.path.splitext(os.path.basename(vector_path))[0]
    tile_index = geopandas.read_file(vector_path, layer=layer)
    # In this case tile_id is the row, and since row numbering starts at 0 but tiles at 1, subtract 1
    # TODO: attribute match the value
    tile_parts["geom_orig"] = tile_index.iloc[(tile_id-1):tile_id]
    tile_parts["geom_orig_buffered"] = tile_parts["geom_orig"]["geometry"].buffer(buffer)
    tile_parts["bbox_orig"] = tile_parts["geom_orig"].bounds.iloc[0].to_list()
    tile_parts["tile_crs"] = CRS.from_wkt(tile_index.crs.to_wkt()) #A rasterio CRS object

    # Properties of 4326 version of tile
    tile_parts["geom_4326"] = tile_parts["geom_orig"].to_crs(4326)
    tile_parts["bbox_4326"] = tile_parts["geom_4326"].bounds.iloc[0].to_list()
    tile_parts["geom_4326_buffered"] =  tile_parts["geom_orig_buffered"].to_crs(4326)
    tile_parts["bbox_4326_buffered"] = tile_parts["geom_4326_buffered"].bounds.iloc[0].to_list()
    
    return tile_parts

def maap_search_get_h5_list(tile_num, tile_fn="/projects/maap-users/alexdevseed/boreal_tiles.gpkg", layer="boreal_tiles_albers",DATE_START='06-01', DATE_END='09-30', YEARS=[2019, 2020, 2021], version=4):
    '''
    Return a list of ATL08 h5 names that intersect a tile for a give date range across a set of years
    '''
    tile_id = get_index_tile(tile_fn, tile_num, buffer=0, layer = layer)

    in_bbox = ",".join(str(coord) for coord in tile_id['bbox_4326'])
    
    print("\tTILE_NUM: {} ({})".format(tile_num, in_bbox) )
    
    out_crs = tile_id['tile_crs']
    
    DATE_START = DATE_START + 'T00:00:00Z' # SUMMER start
    DATE_END = DATE_END + 'T23:59:59Z' # SUMMER end
    
    date_filters = [f'{year}-{DATE_START},{year}-{DATE_END}' for year in YEARS]
    version = str(f'{version:03}')
    
    base_query = {
    'short_name':"ATL08",
    'version':version, 
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
    out_h5_list = [os.path.basename(x) for x in granules]
    
    print("\t\t# ATL08 for tile {}: {}".format(tile_num, len(out_h5_list)) )
    
    return(out_h5_list)

def extract_value_gdf(r_fn, pt_gdf, bandnames: list, reproject=True, TEST=False):
    """Extract raster band values to the obs of a geodataframe
    """

    print("\tExtracting raster values from: ", r_fn)
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
        
        #print('\tDataframe has new raster value column: {}'.format(bandname))
        r = None
        
    r_src.close()
    
    print('\tReturning {} points with {} new raster value columns: {}'.format(len(pt_gdf), len(bandnames), bandnames))
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

def add_neighbors_gdf(input_gdf, input_id_field):
    input_gdf["NEIGHBORS"] = None
    for index, feature in input_gdf.iterrows():   
        #print(tile.tile_num)
        # get 'not disjoint' countries
        neighbors = input_gdf[~input_gdf.geometry.disjoint(feature.geometry)][input_id_field].tolist()
        #print(feature.tile_num)
        # remove own name of the country from the list
        neighbors = [ fid for fid in neighbors if feature[input_id_field] != fid ]
        #print(neighbors)

        # https://stackoverflow.com/questions/57348503/how-do-you-store-a-tuple-in-a-geopandas-geodataframe
        input_gdf['NEIGHBORS'] = input_gdf.apply(lambda row: (neighbors), axis=1)
        
    return input_gdf

def get_neighbors(input_gdf, input_id_field, input_id):
    for index, feature in input_gdf.iterrows():   
        if feature[input_id_field] == input_id:
            #print(tile.tile_num)
            # get 'not disjoint' countries
            neighbors = input_gdf[~input_gdf.geometry.disjoint(feature.geometry)][input_id_field].tolist()
            #print(feature.tile_num)
            # remove own name of the country from the list
            neighbors = [ fid for fid in neighbors if feature[input_id_field] != fid ]
            #print(neighbors)

            # add names of neighbors as NEIGHBORS value
            #boreal_tile_index.at[index, "NEIGHBORS"] = ", ".join(neighbors)
            if False:
                # This will put the neighbors list into a field in the subset df
                # https://stackoverflow.com/questions/57348503/how-do-you-store-a-tuple-in-a-geopandas-geodataframe
                subset_df = input_gdf.loc[input_gdf[input_id_field] == input_id]
                subset_df["NEIGHBORS"] = None
                subset_df['NEIGHBORS'] = subset_df.apply(lambda row: (neighbors), axis=1)

    
    return neighbors