#! /usr/bin/env python

import json
import os
import sys
import glob
import time

import itertools

import pandas as pd
from pyproj import CRS, Transformer

import numpy
import geopandas

import argparse

from maap.maap import MAAP
maap = MAAP()

import FilterUtils
import ExtractUtils

import csv

def get_atl08_csv_list(dps_dir_csv, seg_str, csv_list_fn, col_name='local_path'):
    print(dps_dir_csv + "/**/ATL08*" + seg_str + ".csv") 
    #seg_str="_30m"
    print('Running glob.glob to return a list of csv paths...')
    all_atl08_csvs = glob.glob(dps_dir_csv + "/**/ATL08*" + seg_str + ".csv", recursive=True)
    print(len(all_atl08_csvs))
    all_atl08_csvs_df = pd.DataFrame({col_name: all_atl08_csvs})
    all_atl08_csvs_df.to_csv(csv_list_fn)
    return(all_atl08_csvs_df)

def get_stack_fn(stack_list_fn, in_tile_num, col_name='local_path'):
    # Find most recent topo/Landsat stack path for tile in list of stack paths from *tindex_master.csv
    # *tindex_master.csv made with CountOutput.py
    all_stacks_df = pd.read_csv(stack_list_fn)
    
    # Get the s3 location from the location (local_path) indicated in the tindex master csv
    all_stacks_df['s3'] = [local_to_s3(local_path, args.user) for local_path in all_stacks_df[col_name]]
    
    stack_for_tile = all_stacks_df[all_stacks_df[col_name].str.contains("_"+str(in_tile_num)+"_")]
    print("\nGetting stack fn from: ", stack_list_fn)
    [print("\t",i) for i in stack_for_tile[col_name].to_list()]
    stack_for_tile_fn = stack_for_tile[col_name].to_list()[0]
    if len(stack_for_tile)==0:
        stack_for_tile_fn = None
    return(stack_for_tile_fn)

def local_to_s3(url, user='nathanmthomas'):
    ''' A Function to convert local paths to s3 urls'''
    return url.replace('/projects/my-private-bucket', f's3://maap-ops-workspace/{user}')

def main():
    '''
    tile_atl08.py: By tile, query, filter, and extract covariates for ATL08
    1. Query MAAP by tile bounds to find all intersecting ATL08
    2. Read a large list of extracted ATL08 csv paths
    3. Find the ATL08 csvs from extract_atl08 that are associated with the ATL08 granules that intersect this tile
    4. Merge all ATL08 CSV files for the current tile into a pandas df
    5. Filter ATL08 by tile bounds
    6. Filter ATL08 by quality
    7. Convert ATL08 df to geodataframe
    8. Extract Topo vars to ATL08
    9. Extract Landsat vars tp ATL08
    10. Write ATL08 filtered df with extracted covars to CSV and GeoJSON
    
    Example call:
    python tile_atl08.py -o /projects/my-public-bucket/atl08_filt_covar_tiles -csv_list_fn /projects/my-public-bucket/DPS_tile_lists/TEST_BUILD_extract_atl08_csv_list.csv --do_30m -in_tile_num 3000 --extract_covars
    
    python tile_atl08.py -o /projects/my-public-bucket/atl08_filt_covar_tiles -csv_list_fn /projects/shared-buckets/lduncanson/DPS_tile_lists/ATL08_tindex_master.csv --do_30m -in_tile_num 3000 --extract_covars
    '''
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("-ept", "--in_ept_fn", type=str, help="The input ept of ATL08 observations") 
    parser.add_argument("-in_tile_num", type=int, help="The id number of an input vector tile that will define the bounds for ATL08 subset")
    parser.add_argument("-in_tile_fn", type=str, default="/projects/shared-buckets/nathanmthomas/boreal_grid_albers90k_gpkg.gpkg", help="The input filename of a set of vector tiles that will define the bounds for ATL08 subset")
    parser.add_argument("-in_tile_layer", type=str, default="grid_boreal_albers90k_gpkg", help="The layer name of the stack tiles dataset")
    parser.add_argument("-csv_list_fn", type=str, default=None, help="The file of all CSVs paths")
    parser.add_argument("-topo_stack_list_fn", type=str, default="/projects/shared-buckets/nathanmthomas/DPS_tile_lists/Topo_tindex_master.csv", help="The file of all topo stack paths")
    parser.add_argument("-landsat_stack_list_fn", type=str, default="/projects/shared-buckets/nathanmthomas/DPS_tile_lists/Landsat_tindex_master.csv", help="The file of all Landsat stack paths")
    parser.add_argument("-user", type=str, default="nathanmthomas", help="MAAP username for completing the s3 path to tindex master csvs")
    #parser.add_argument("--local", dest='local', action='store_true', help="Dictate whether landsat covars is a run using local paths")
    #parser.set_defaults(local=False)
    #parser.add_argument("-t_h_can", "--thresh_h_can", type=int, default=100, help="The threshold height below which ATL08 obs will be returned")
    #parser.add_argument("-t_h_dif", "--thresh_h_dif", type=int, default=100, help="The threshold elev dif from ref below which ATL08 obs will be returned")
    #parser.add_argument("-m_min", "--month_min", type=int, default=6, help="The min month of each year for which ATL08 obs will be used")
    #parser.add_argument("-m_max", "--month_max", type=int, default=9, help="The max month of each year for which ATL08 obs will be used")
    #parser.add_argument('-ocl', '--out_cols_list', nargs='+', default=[], help="A select list of strings matching ATL08 col names from the input EPT that will be returned in a pandas df after filtering and subsetting")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The output dir of the filtered and subset ATL08 csv")
    parser.add_argument("-dps_dir_csv", type=str, default=None, help="The top-level DPS output dir for the ATL08 csv files (needed if csv_list_fn doesnt exist)")
    parser.add_argument("-date_start", type=str, default="06-01", help="Seasonal start MM-DD")
    parser.add_argument("-date_end", type=str, default="09-30", help="Seasonal end MM-DD")
    #parser.add_argument('--maap_query', dest='maap_query', action='store_true', help='Run a MAAP query by tile to return list of ATL08 h5 that forms the database of ATL08 observations')
    #parser.set_defaults(maap_query=False)
    parser.add_argument('--do_30m', dest='do_30m', action='store_true', help='Turn on 30m ATL08 extraction')
    parser.set_defaults(do_30m=False)
    parser.add_argument('--extract_covars', dest='extract_covars', action='store_true', help='Do extraction of covars for each ATL08 obs')
    parser.set_defaults(extract_covars=False)
    parser.add_argument('--TEST', dest='TEST', action='store_true', help='Do testing')
    parser.set_defaults(TEST=False)
    parser.add_argument('--DEBUG', dest='DEBUG', action='store_true', help='Do debugging')
    parser.set_defaults(DEBUG=False)
    
    args = parser.parse_args()
    #if args.in_ept_fn == None and not args.maap_query:
    #    print("The flag 'maap_query' is false so you need an input filename of the EPT database of ATL08 obs tiles that will be quality-filtered and subset by tile")
    #    os._exit(1)
    if args.in_tile_fn == None:
        print("Input a filename of the vector tiles that represents the arrangement by which the ATL08 obs will be organized")
        os._exit(1)
    elif args.in_tile_num == None:
        print("Input a specific tile id from the vector tiles the organize the ATL08 obs")
        os._exit(1)
    elif args.in_tile_layer == None:
        print("Input a layer name from the tile vector file")
        os._exit(1)   
      
    #in_ept_fn = args.in_ept_fn
    in_tile_num = args.in_tile_num
    in_tile_fn = args.in_tile_fn
    in_tile_layer = args.in_tile_layer
    csv_list_fn = args.csv_list_fn
    topo_stack_list_fn = args.topo_stack_list_fn
    landsat_stack_list_fn = args.landsat_stack_list_fn
    date_start = args.date_start
    date_end = args.date_end
    #thresh_h_can = args.thresh_h_can
    #thresh_h_dif = args.thresh_h_dif
    #month_min = args.month_min
    #month_max = args.month_max
    #out_cols_list = args.out_cols_list
    output_dir = args.output_dir
    do_30m = args.do_30m
    dps_dir_csv = args.dps_dir_csv
    user = args.user
    
    in_tile_num = int(in_tile_num)
    print("\nWorking on tile: ", in_tile_num)
    print("From layer: ", in_tile_layer)
    print("In vector file: ", in_tile_fn)
    
    out_name_stem = "atl08_filt"
    cur_date = time.strftime("%Y%m%d") #"%Y%m%d%H%M%S"
    #cur_date = "20210819"
    
    # TODO: make this an arg
    years_list = [2018, 2019, 2020, 2021]
    
    seg_str = '_100m'
    if do_30m:
        seg_str = '_30m'
    if args.TEST:
        seg_str = '' 
    
    if csv_list_fn is not None:
        
        print("\nDoing MAAP query by tile bounds to find all intersecting ATL08 ")
        # Get a list of all ATL08 H5 granule names intersecting the tile (this will be a small list)
        all_atl08_for_tile = ExtractUtils.maap_search_get_h5_list(tile_num=in_tile_num, tile_fn=in_tile_fn, layer=in_tile_layer, DATE_START=date_start, DATE_END=date_end, YEARS=years_list)
        
        # Print ATL08 h5 granules for tile
        #print([os.path.basename(f) for f in all_atl08_for_tile])      
        
        if not os.path.isfile(csv_list_fn):
            print("\nNo CSV list of extracted ATL08 csvs exist.")
            print("Build one now. This takes a long time.")
            if dps_dir_csv is None:
                print("To build one, you need to specify a top-level DPS output dir under which the extracted ATL08 csvs can be found.")
                print("Go find this dir, and specify it as the arg to -dps_dir_csv")
                os._exit(1)
            all_atl08_csvs_df = get_atl08_csv_list(dps_dir_csv, seg_str, csv_list_fn)
        else:
            print(f"\nReading existing list of ATL08 CSVs: {csv_list_fn}")
            print("\tDoing 30m ATL08 data? ", do_30m)
            all_atl08_csvs_df = pd.read_csv(csv_list_fn)
            
        # Get the s3 location from the location (local_path) indicated in the tindex master csv
        all_atl08_csvs_df['s3'] = [local_to_s3(local_path, args.user) for local_path in all_atl08_csvs_df['local_path']] # in earlier versions of the atl08 csv list 'local_path' was called 'path'
            
        # Find the ATL08 CSVs from extract that are associated with the ATL08 granules that intersect this tile
        # These CSvs are nested deep after DPS runs
        # They should match all the ATL08 granules, but probably wont bc: (1) did DPS for ATL08 30m fully complete with no fails? (2) did DPS for extract fully complete with no fails?
        all_atl08_csvs_FOUND, all_atl08_csvs_NOT_FOUND = FilterUtils.find_atl08_csv_tile(all_atl08_for_tile, all_atl08_csvs_df, seg_str, col_name='s3', DEBUG=args.DEBUG) 
        
        #all_atl08_csvs_FOUND = [x for x in all_atl08_h5_with_csvs_for_tile if x not in all_atl08_csvs_NOT_FOUND]
        print("\t# of ATL08 CSV found for tile {}: {}".format(in_tile_num, len(all_atl08_csvs_FOUND)))
        print("\t# of ATL08 CSV NOT found for tile {}: {}".format(in_tile_num, len(all_atl08_csvs_NOT_FOUND)))
        if len(all_atl08_csvs_FOUND) == 0:
            print('\tNo ATL08 extracted for this tile.')
            os._exit(1)
        # Merge all ATL08 CSV files for the current tile into a pandas df
        print("Creating pandas data frame...")
        atl08 = pd.concat([pd.read_csv(f) for f in all_atl08_csvs_FOUND ], sort=False, ignore_index=True)
        atl08 = FilterUtils.prep_filter_atl08_qual(atl08)

        print("\nFiltering by tile: {}".format(in_tile_num))
        # Get tile bounds as xmin,xmax,ymin,ymax
        tile = ExtractUtils.get_index_tile(in_tile_fn, in_tile_num, buffer=0, layer=in_tile_layer)
        in_bounds = FilterUtils.reorder_4326_bounds(tile)
        print(in_bounds)
        
        # Now filter ATL08 obs by tile bounds (changed to a geopandas clip instead of simple bounds approach)
        atl08 = FilterUtils.filter_atl08_bounds_clip(atl08, tile['geom_4326'])

    else:
        print("\nNo CSV fn of paths to all extracted ATL08 csvs dir specified.")
        print("Need to get ATL08 CSV list to match with tile bound results from MAAP query.")
        print("Exiting...\n")

    
    # Filter by quality
    atl08_pdf_filt = FilterUtils.filter_atl08_qual(atl08, SUBSET_COLS=True, DO_PREP=False,
                                                       subset_cols_list=['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can','h_max_can','seg_landcov'], 
                                                       filt_cols=['h_can','h_dif_ref','m','msw_flg','beam_type','seg_snow'], 
                                                       thresh_h_can=100, thresh_h_dif=100, month_min=6, month_max=9)
    atl08=None
    
    # Convert to geopandas data frame in lat/lon
    atl08_gdf = geopandas.GeoDataFrame(atl08_pdf_filt, geometry=geopandas.points_from_xy(atl08_pdf_filt.lon, atl08_pdf_filt.lat), crs='epsg:4326')
    atl08_pdf_filt=None
    
    if args.extract_covars and len(atl08_gdf) > 0:

        topo_covar_fn    = get_stack_fn(topo_stack_list_fn, in_tile_num, col_name='location') # will need to change col_name to 'local_path' soon
        landsat_covar_fn = get_stack_fn(landsat_stack_list_fn, in_tile_num, col_name='location') # will need to change col_name to 'local_path' soon
       
        if topo_covar_fn is not None and landsat_covar_fn is not None:
            print(f'\nExtract covars for {len(atl08_gdf)} ATL08 obs...')
            
            atl08_gdf = ExtractUtils.extract_value_gdf(topo_covar_fn, atl08_gdf, ["elevation","slope","tsri","tpi", "slopemask"], reproject=True)
            out_name_stem = out_name_stem + "_topo"

            atl08_gdf = ExtractUtils.extract_value_gdf(landsat_covar_fn, atl08_gdf, ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'ValidMask', 'Xgeo', 'Ygeo'], reproject=False)
            out_name_stem = out_name_stem + "_landsat"
        
    if len(atl08_gdf) == 0:
        print(f"No ATL08 obs. for tile {in_tile_num}")
    else:
        # CSV/geojson the file
        out_fn = os.path.join(output_dir, out_name_stem + "_" + str(cur_date) + "_" + str(in_tile_num))
        
        atl08_gdf.to_csv(out_fn+".csv", index=False, encoding="utf-8-sig")
        atl08_gdf.to_file(out_fn+'.geojson', driver="GeoJSON")

        print("Wrote output csv/geojson of filtered ATL08 obs with topo and Landsat covariates for tile {}: {}".format(in_tile_num, out_fn) )

if __name__ == "__main__":
    main()