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
maap = MAAP(maap_host='api.ops.maap-project.org')

import FilterUtils
import ExtractUtils

import csv
import fsspec
import s3fs

from shutil import copy

def get_atl08_csv_list(dps_dir_csv, seg_str, csv_list_fn, col_name='local_path'):
    print(dps_dir_csv + "/**/ATL08*" + seg_str + ".csv") 
    #seg_str="_30m"
    print('Running glob.glob to return a list of csv paths...')
    all_atl08_csvs = glob.glob(dps_dir_csv + "/**/ATL08*" + seg_str + ".csv", recursive=True)
    print(len(all_atl08_csvs))
    all_atl08_csvs_df = pd.DataFrame({col_name: all_atl08_csvs})
    all_atl08_csvs_df.to_csv(csv_list_fn)
    return(all_atl08_csvs_df)

def get_stack_fn(stack_list_fn, in_tile_num, user, col_name='local_path', return_s3=True):
    # Find most recent topo/Landsat stack path for tile in list of stack paths from *tindex_master.csv
    # *tindex_master.csv made with CountOutput.py
    all_stacks_df = pd.read_csv(stack_list_fn)
    
    # Get the s3 location from the location (local_path) indicated in the tindex master csv
    all_stacks_df['s3'] = [local_to_s3(local_path, user) for local_path in all_stacks_df[col_name]]
    
    if return_s3:
        col_name = 's3'
    
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

def update_atl08_version(atl08_h5_name, update_v_str, maap_v_str='003'):
    ''' A function to replace the version string in the ATL08 h5 filename returned from maap query '''
    return atl08_h5_name.replace('_'+maap_v_str+'_', '_'+update_v_str+'_')

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
    
    python tile_atl08.py -o /projects/my-public-bucket/atl08_filt_covar_tiles -csv_list_fn /projects/shared-buckets/lduncanson/DPS_tile_lists/ATL08_tindex_master.csv --do_30m -in_tile_num 3000 --extract_covars -years_list 2020
    
    # Norway tile test that will return sol_el to the tiled ATL08 dataset
    python tile_atl08.py -o /projects/my-public-bucket/atl08_filt_covar_tiles -csv_list_fn /projects/shared-buckets/lduncanson/DPS_tile_lists/ATL08_tindex_master.csv --do_30m -in_tile_num 224 --extract_covars -years_list 2018 2019 2020 -thresh_sol_el 5 -v_ATL08 4
    
    python tile_atl08.py -o /projects/my-public-bucket/atl08_filt_covar_tiles -csv_list_fn /projects/shared-buckets/lduncanson/DPS_tile_lists/ATL08_tindex_master.csv --do_30m -in_tile_num 222 --extract_covars -years_list 2020 -thresh_sol_el 5 -v_ATL08 4 -minmonth 6 -maxmonth 9
    
    # Grabbed the echo'd call from a successful DPS run
    # removed the path to the script ; removed --do_dps ; replace the gpkg path ; replaced the output dir ; changed input tile list paths from s3 path to local path
    python tile_atl08.py --updated_filters --extract_covars --do_30m -years_list 2020 -o /projects/my-public-bucket/atl08_filt_covar_tiles -in_tile_num 132 -in_tile_fn /projects/shared-buckets/nathanmthomas/boreal_tiles_v002.gpkg -in_tile_layer boreal_tiles_v002 -csv_list_fn /projects/shared-buckets/lduncanson/DPS_tile_lists/ATL08_tindex_master.csv -topo_stack_list_fn /projects/shared-buckets/nathanmthomas/DPS_tile_lists/Topo_tindex_master.csv -landsat_stack_list_fn /projects/shared-buckets/nathanmthomas/DPS_tile_lists/Landsat_tindex_master.csv -user_stacks nathanmthomas -user_atl08 lduncanson -thresh_sol_el 5 -v_ATL08 4 -minmonth 6 -maxmonth 9
    
    Run on v4; need to make sure you specify a cols list that is present in what extract atl08 returned (so, no h_can_unc, seg_cover, )
    for v4 data extracted before late Dec 2021 updates, use these:
    atl08_cols_list = ['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can','h_max_can', 'ter_slp', 'seg_landcov', 'sol_el']
    for v4 data extracted after updates to extract_atl08.py (jan 2022), use these:
    atl08_cols_list = ['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can', 'h_max_can', 'ter_slp', 'seg_landcov', 'sol_el', 'h_can_unc', 'h_te_best', 'h_te_unc', 'h_dif_ref']
    for v5 data, all extracts will be run after the jan 2022 updates, and the seg_cover field is newly available, so use these:
    atl08_cols_list = ['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can', 'h_max_can', 'ter_slp', 'seg_landcov', 'sol_el', 'h_can_unc', 'h_te_best', 'h_te_unc', 'h_dif_ref', 'seg_cover']
    
    previous default tiles: "/projects/shared-buckets/nathanmthomas/boreal_grid_albers90k_gpkg.gpkg"
    previous layername: grid_boreal_albers90k_gpkg
    '''
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("-ept", "--in_ept_fn", type=str, help="The input ept of ATL08 observations") 
    parser.add_argument("-in_tile_num", type=int, help="The id number of an input vector tile that will define the bounds for ATL08 subset")
    parser.add_argument("-in_tile_fn", type=str, default="/projects/shared-buckets/nathanmthomas/boreal_tiles_v003.gpkg", help="The input filename of a set of vector tiles that will define the bounds for ATL08 subset")
    parser.add_argument("-in_tile_layer", type=str, default="boreal_tiles_v003", help="The layer name of the stack tiles dataset")
    parser.add_argument("-in_tile_id_col", type=str, default="tile_num", help="The column of the tile layer name of the stack tiles dataset that holds the tile num")
    parser.add_argument("-csv_list_fn", type=str, default=None, help="The file of all CSVs paths")
    parser.add_argument("-topo_stack_list_fn", type=str, default="/projects/shared-buckets/nathanmthomas/DPS_tile_lists/Topo_tindex_master.csv", help="The file of all topo stack paths")
    parser.add_argument("-landsat_stack_list_fn", type=str, default="/projects/shared-buckets/nathanmthomas/DPS_tile_lists/Landsat_tindex_master.csv", help="The file of all Landsat stack paths")
    parser.add_argument("-user_stacks", type=str, default="nathanmthomas", help="MAAP username for the workspace in which the stacks were built; will help complete the s3 path to tindex master csvs")
    parser.add_argument("-user_atl08", type=str, default="lduncanson", help="MAAP username for the workspace in which the ATL08 csvs were extracted; will help complete the s3 path to tindex master csvs")
    #parser.add_argument("--local", dest='local', action='store_true', help="Dictate whether landsat covars is a run using local paths")
    #parser.set_defaults(local=False)
    #parser.add_argument("-t_h_can", "--thresh_h_can", type=int, default=100, help="The threshold height below which ATL08 obs will be returned")
    #parser.add_argument("-t_h_dif", "--thresh_h_dif", type=int, default=100, help="The threshold elev dif from ref below which ATL08 obs will be returned")
    #parser.add_argument("-m_min", "--month_min", type=int, default=6, help="The min month of each year for which ATL08 obs will be used")
    #parser.add_argument("-m_max", "--month_max", type=int, default=9, help="The max month of each year for which ATL08 obs will be used")
    parser.add_argument("-minmonth" , type=int, default=6, help="Min month of ATL08 shots for output to include")
    parser.add_argument("-maxmonth" , type=int, default=9, help="Max month of ATL08 shots for output to include")
    parser.add_argument("-thresh_sol_el", type=int, default=0, help="Threshold for sol elev for obs of interest")
    parser.add_argument('-atl08_cols_list', nargs='+', default=['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can','h_max_can', 'ter_slp','h_te_best', 'seg_landcov','sol_el','y','m','doy'], help="A select list of strings matching ATL08 col names that will be returned in a pandas df after filtering and subsetting")
    parser.add_argument('-topo_cols_list', nargs='+',  default=["elevation","slope","tsri","tpi", "slopemask"], help='Topo vars to extract')
    parser.add_argument('-landsat_cols_list', nargs='+',  default=['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'ValidMask', 'Xgeo', 'Ygeo'], help='Landsat composite vars to extract')
    parser.add_argument("-o", "--outdir", type=str, default=None, help="The output dir of the filtered and subset ATL08 csv")
    parser.add_argument("-dps_dir_csv", type=str, default=None, help="The top-level DPS output dir for the ATL08 csv files (needed if csv_list_fn doesnt exist)")
    #parser.add_argument("-date_start", type=str, default="06-01", help="Seasonal start MM-DD")
    #parser.add_argument("-date_end", type=str, default="09-30", help="Seasonal end MM-DD")
    parser.add_argument('-years_list', nargs='+', type=int, default=[2020], help="Years of ATL08 used")
    parser.add_argument('-v_ATL08', type=int, default=4, help='The version of ATL08 that was extracted from the rebinning. Needed in case version string isnt updated in maap')
    parser.add_argument('-N_OBS_SAMPLE', type=int, default=250, help='Number of ATL08 obs to include in the sample CSV for the tile.')
    #parser.add_argument('-to_dir_cog', type=str, default='/projects/my-public-bucket/in_stacks_copy', help='COG copies of input stacks that are accessible with R in a workspace other than that of their creation')
    #parser.add_argument('--maap_query', dest='maap_query', action='store_true', help='Run a MAAP query by tile to return list of ATL08 h5 that forms the database of ATL08 observations')
    #parser.set_defaults(maap_query=False)
    parser.add_argument('--do_30m', dest='do_30m', action='store_true', help='Turn on 30m ATL08 extraction')
    parser.set_defaults(do_30m=False)
    parser.add_argument('--extract_covars', dest='extract_covars', action='store_true', help='Do extraction of covars for each ATL08 obs')
    parser.set_defaults(extract_covars=False)
    parser.add_argument('--do_dps', dest='do_dps', action='store_true', help='Do this as a DPS job')
    parser.set_defaults(do_dps=False)
    parser.add_argument('--TEST', dest='TEST', action='store_true', help='Do testing')
    parser.set_defaults(TEST=False)
    parser.add_argument('--DEBUG', dest='DEBUG', action='store_true', help='Do debugging')
    parser.set_defaults(DEBUG=False)
    parser.add_argument('--updated_filters', dest='updated_filters', action='store_true', help='Use updated quality filtering applied to ATL08 from FilterUtils')
    parser.set_defaults(updated_filters=False)

    
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

    years_list = args.years_list
    v_ATL08 = args.v_ATL08
    #thresh_h_can = args.thresh_h_can
    #thresh_h_dif = args.thresh_h_dif
    minmonth = args.minmonth
    maxmonth = args.maxmonth
    #date_start = args.date_start
    #date_end = args.date_end
    
    startday = 1
    endday = 31
    if maxmonth in [4,6,9,11]:
        endday = 30
    if maxmonth in [2]:
        endday = 28

    date_start = str(f'{minmonth:02}-{startday:02}')
    date_end = str(f'{maxmonth:02}-{endday:02}')
    
    thresh_sol_el = args.thresh_sol_el
    atl08_cols_list = args.atl08_cols_list
    topo_cols_list = args.topo_cols_list
    landsat_cols_list = args.landsat_cols_list
    outdir = args.outdir
    do_30m = args.do_30m
    dps_dir_csv = args.dps_dir_csv
    updated_filters = args.updated_filters
    N_OBS_SAMPLE = args.N_OBS_SAMPLE

    DEBUG = args.DEBUG
    
    seg_str = '_100m'
    if do_30m:
        seg_str = '_30m'
    if args.TEST:
        seg_str = '' 
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    in_tile_num = int(in_tile_num)
    print("\nWorking on tile:\t", in_tile_num)
    print("From layer:\t\t", in_tile_layer)
    print("In vector file:\t\t", in_tile_fn)
    print("ATL08 version:\t\t", v_ATL08)
    print("Season start:\t\t", date_start)
    print("Season end:\t\t", date_end)
    print("Years:\t\t\t", years_list)
    print("ATL08 bin length:\t",seg_str.replace("_",''))
    
    out_name_stem = "atl08_"+str(f'{v_ATL08:03}')+seg_str+"_filt"
    cur_date = time.strftime("%Y%m%d") #"%Y%m%d%H%M%S"  
    
    if csv_list_fn is not None:
        
        print("\nDoing MAAP query by tile bounds to find all intersecting ATL08 ")
        # Get a list of all ATL08 H5 granule names intersecting the tile (this will be a small list)
        all_atl08_for_tile = ExtractUtils.maap_search_get_h5_list(tile_num=in_tile_num, id_col=args.in_tile_id_col, tile_fn=in_tile_fn, layer=in_tile_layer, DATE_START=date_start, DATE_END=date_end, YEARS=years_list, version=v_ATL08)
         
        if DEBUG:
            # Print ATL08 h5 granules for tile
            print([os.path.basename(f) for f in all_atl08_for_tile])      
        
        if not os.path.isfile(csv_list_fn) and not args.do_dps:
            print("\nThis is not a DPS job")
            print("\nNo CSV list of extracted ATL08 csvs exist.")
            print("Build one now. This takes a long time.")
            if dps_dir_csv is None:
                print("To build one, you need to specify a top-level DPS output dir under which the extracted ATL08 csvs can be found.")
                print("Go find this dir, and specify it as the arg to -dps_dir_csv")
                os._exit(1)
            all_atl08_csvs_df = get_atl08_csv_list(dps_dir_csv, seg_str, csv_list_fn)
        else:
            print("\nThis is either a DPS job (for which the CSV has an s3 path, or the CSV exists locally.)")
            print(f"\nReading existing list of ATL08 CSVs: {csv_list_fn}")
            all_atl08_csvs_df = pd.read_csv(csv_list_fn)
            
        print("\tDoing 30m ATL08 data? ", do_30m) 
        
        # Get the s3 location from the location (local_path) indicated in the tindex master csv
        all_atl08_csvs_df['s3'] = [local_to_s3(local_path, args.user_atl08) for local_path in all_atl08_csvs_df['local_path']] # in earlier versions of the atl08 csv list 'local_path' was called 'path'
        if DEBUG:
            print(all_atl08_csvs_df['s3'][0])
            
        # Find the ATL08 CSVs from extract that are associated with the ATL08 granules that intersect this tile
        # These CSvs are nested deep after DPS runs
        # They should match all the ATL08 granules, but probably wont bc: (1) did DPS for ATL08 30m fully complete with no fails? (2) did DPS for extract fully complete with no fails?
        all_atl08_csvs_FOUND, all_atl08_csvs_NOT_FOUND = FilterUtils.find_atl08_csv_tile(all_atl08_for_tile, all_atl08_csvs_df, seg_str, col_name='s3', DEBUG=DEBUG) 
        
        #all_atl08_csvs_FOUND = [x for x in all_atl08_h5_with_csvs_for_tile if x not in all_atl08_csvs_NOT_FOUND]
        print("\t# of ATL08 CSV found for tile {}: {}".format(in_tile_num, len(all_atl08_csvs_FOUND)))
        print("\t# of ATL08 CSV NOT found for tile {}: {}".format(in_tile_num, len(all_atl08_csvs_NOT_FOUND)))
        if len(all_atl08_csvs_FOUND) == 0:
            print('\tNo ATL08 extracted for this tile.')
            os._exit(1)
        # Merge all ATL08 CSV files for the current tile into a pandas df
        print("Creating pandas data frame...")
        atl08 = pd.concat([pd.read_csv(f) for f in all_atl08_csvs_FOUND ], sort=False, ignore_index=True)
        if DEBUG:
            atl08.to_csv(os.path.join(outdir, "atl08_all_" + str(cur_date) + "_" + str(f'{in_tile_num:04}.csv')))
            print(atl08.info())
        atl08 = FilterUtils.prep_filter_atl08_qual(atl08)

        print("\nFiltering by tile: {}".format(in_tile_num))
        # Get tile bounds as xmin,xmax,ymin,ymax
        tile = ExtractUtils.get_index_tile(vector_path=in_tile_fn, id_col=args.in_tile_id_col, tile_id=in_tile_num, buffer=0, layer=in_tile_layer)
        in_bounds = FilterUtils.reorder_4326_bounds(tile)
        print(in_bounds)
        
        # Now filter ATL08 obs by tile bounds (changed to a geopandas clip instead of simple bounds approach)
        atl08 = FilterUtils.filter_atl08_bounds_clip(atl08, tile['geom_4326'])

    else:
        print("\nNo CSV fn of paths to all extracted ATL08 csvs dir specified.")
        print("Need to get ATL08 CSV list to match with tile bound results from MAAP query.")
        print("Exiting...\n")

    
    # Filter by quality
    if v_ATL08 == 4:
        print(f'\nATL08 version is {v_ATL08}. Cannot apply aggressive land-cover filtering.')
        updated_filters = False
        
    if not updated_filters:
        '''print('Original quality filtering')
        atl08_pdf_filt = FilterUtils.filter_atl08_qual(atl08, SUBSET_COLS=True, DO_PREP=False,
                                                           subset_cols_list=atl08_cols_list, #['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can','h_max_can','seg_landcov','night_flg'], 
                                                           filt_cols=['h_can','h_dif_ref','m','msw_flg','beam_type','seg_snow'], 
                                                           thresh_h_can=100, thresh_h_dif=100, month_min=minmonth, month_max=maxmonth)
                                                           '''
        print('Quality filtering with thresholding use in the preliminary map ...')
        atl08_pdf_filt = FilterUtils.filter_atl08_qual_v2(atl08, SUBSET_COLS=True, DO_PREP=False,
                                                           #subset_cols_list=['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can','h_max_can','seg_landcov','night_flg'], 
                                                           subset_cols_list = atl08_cols_list,
                                                           filt_cols=['h_can','h_dif_ref','m','msw_flg','beam_type','seg_snow','sig_topo'], 
                                                           thresh_h_can=100, thresh_h_dif=25, thresh_sig_topo=2.5, month_min=minmonth, month_max=maxmonth)
    else:  
        print('Quality filtering with aggressive land-cover based (v3) filters updated in Jan/Feb 2022 ...')
        atl08_pdf_filt = FilterUtils.filter_atl08_qual_v3(atl08, SUBSET_COLS=True, DO_PREP=False,
                                                          subset_cols_list = atl08_cols_list + ['seg_cover', 'granule_name'], 
                                                   filt_cols=['h_can','h_dif_ref','m','msw_flg','beam_type','seg_snow','sig_topo'], 
                                                   list_lc_h_can_thresh=args.list_lc_h_can_thresh,
                                                   thresh_h_can=100, thresh_h_dif=25, thresh_sig_topo=2.5, month_min=minmonth, month_max=maxmonth)
        
    atl08=None
    
    # Convert to geopandas data frame in lat/lon
    atl08_gdf = geopandas.GeoDataFrame(atl08_pdf_filt, geometry=geopandas.points_from_xy(atl08_pdf_filt.lon, atl08_pdf_filt.lat), crs='epsg:4326')
    atl08_pdf_filt=None
    
    if args.extract_covars and len(atl08_gdf) > 0:

        topo_covar_fn    = get_stack_fn(topo_stack_list_fn, in_tile_num, user=args.user_stacks, col_name='local_path') 
        landsat_covar_fn = get_stack_fn(landsat_stack_list_fn, in_tile_num, user=args.user_stacks, col_name='local_path')
       
        if topo_covar_fn is not None and landsat_covar_fn is not None:
            print(f'\nExtract covars for {len(atl08_gdf)} ATL08 obs...')
            
            atl08_gdf = ExtractUtils.extract_value_gdf(topo_covar_fn, atl08_gdf, topo_cols_list, reproject=True)
            out_name_stem = out_name_stem + "_topo"

            atl08_gdf = ExtractUtils.extract_value_gdf(landsat_covar_fn, atl08_gdf, landsat_cols_list, reproject=False)
            out_name_stem = out_name_stem + "_landsat"
        
    if len(atl08_gdf) == 0:
        print(f"No ATL08 obs. for tile {in_tile_num}")
    else:
        # CSV/geojson the file
        out_fn = os.path.join(outdir, out_name_stem + "_" + str(cur_date) + "_" + str(f'{in_tile_num:04}'))
        
        print(f"Writing a the tile's CSV with extracted covars: \n{out_fn}.csv")
        atl08_gdf.to_csv(out_fn+".csv", index=False, encoding="utf-8-sig")
        
        if len(atl08_gdf[atl08_gdf.sol_el < thresh_sol_el]) > N_OBS_SAMPLE:
            print(f'Writing a sample CSV of {N_OBS_SAMPLE} sol elev obs. < {thresh_sol_el}: {out_fn+f"_SAMPLE_n{N_OBS_SAMPLE}.csv"}')
            atl08_gdf[atl08_gdf.sol_el < thresh_sol_el].sample(N_OBS_SAMPLE, replace=False).to_csv(out_fn+f"_SAMPLE_n{N_OBS_SAMPLE}.csv", index=False, encoding="utf-8-sig")
        else:
            N_OBS_SAMPLE = len(atl08_gdf[atl08_gdf.sol_el < thresh_sol_el]) 
            print(f'Writing out a CSV of all sol elev obs. < {thresh_sol_el} (no sampling - too few obs.): {out_fn+f"_SAMPLE_n{N_OBS_SAMPLE}.csv"}')
            atl08_gdf[atl08_gdf.sol_el < thresh_sol_el].to_csv(out_fn+f"_SAMPLE_n{N_OBS_SAMPLE}.csv", index=False, encoding="utf-8-sig")

        #atl08_gdf.to_file(out_fn+'.geojson', driver="GeoJSON")

        print("Wrote output csv/geojson of filtered ATL08 obs with topo and Landsat covariates for tile {}: {}".format(in_tile_num, out_fn) )

if __name__ == "__main__":
    main()