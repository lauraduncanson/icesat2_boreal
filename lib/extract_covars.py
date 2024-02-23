import os
import sys
import glob

import pandas as pd
import geopandas as gpd
import s3fs

import argparse

from maap.maap import MAAP
maap = MAAP(maap_host='api.maap-project.org')

import FilterUtils
import ExtractUtils
import CovariateUtils

def extract_covars(s3_atl08_gdf_fn, tindex_fn_list: list, RETURN_DF=False):
    
    '''Extract values from tiled raster covariates to input tiled sets of atl08 observations using the path of the tiled atl08 geodataframe
     + designed to be multiprocessed with a list of tile nums
     + only works when tile nums of ATL08 matches tile nums of raster covariates
     + to work with any polygon, needs to be updated to read in mosaic of all tiled rasters
    '''
    
    s3 = s3fs.S3FileSystem(anon=True)

    print(s3_atl08_gdf_fn)
    
    if s3_atl08_gdf_fn.endswith('parquet'):
        atl08 = gpd.read_parquet(s3_atl08_gdf_fn)
    else:
        atl08 = gpd.read_file(s3_atl08_gdf_fn)
    
    print(atl08.shape)
        
    # Get file name
    tile_num = int(s3_atl08_gdf_fn.split('_')[-1].split('.')[0]) # tile num is last position in '_' delimited filename
    print(tile_num)
    out_atl08_covars_fn = s3_atl08_gdf_fn.split(f'_{tile_num:05}.')[0] + f'_covars_{tile_num:05}.parquet'
    
    if atl08.shape[0] == 0: sys.exit(f'Tile {tile_num:05} has 0 observations. Exiting.')
    
    ###################
    # Extract covariates
    ###################
    print(f'\nExtracting values from {len(tindex_fn_list)} sets of raster stacks and appending as columns to atl08 geodataframe...')
    for tindex_fn in tindex_fn_list: 
        print(tindex_fn)
        covar_fn = CovariateUtils.get_stack_fn(tindex_fn, tile_num, user=None, col_name='local_path')
        # This function probably breaks multiprocessing..prob b/c of s3 session?
        atl08 = ExtractUtils.extract_value_gdf_s3(covar_fn, atl08, None, reproject=True)
        
    atl08.to_parquet(out_atl08_covars_fn)
    print(f'File written:\t{out_atl08_covars_fn}')
    
    if RETURN_DF:
        return atl08
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s3_atl08_gdf_fn", type=str, help="The s3 path to the ATL08 geodataframe used for extraction.")
    parser.add_argument("-tindex_fn_list",  nargs='+', type=str, default=None, help="A list of s3 paths to tindex csv files with s3 paths to raster tiles.")
    parser.add_argument('--RETURN_DF', dest='RETURN_DF', action='store_true', help='Boolean to return a data frame')
    parser.set_defaults(RETURN_DF=False)
    
    args = parser.parse_args()
        
    extract_covars(args.s3_atl08_gdf_fn, args.tindex_fn_list, args.RETURN_DF)
    
    if __name__ == "__main__":
        main()