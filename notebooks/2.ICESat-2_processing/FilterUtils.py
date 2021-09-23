#import pdal
import json
import os

import pandas as pd
import geopandas as gpd

from pyproj import CRS, Transformer

import sys
#sys.path.append('/projects/code/icesat2_boreal/notebooks/3.Gridded_product_development')
#from CovariateUtils import *
import ExtractUtils

#def reorder_4326_bounds(boreal_tile_index_path, test_tile_id, buffer, layer):
def reorder_4326_bounds(tile_parts):    
    #tile_parts = ExtractUtils.get_index_tile(boreal_tile_index_path, test_tile_id, buffer=buffer, layer=layer)
    bounds_order = [0, 2, 1, 3]
    out_4326_bounds = [tile_parts['bbox_4326'][i] for i in bounds_order]
    
    return(out_4326_bounds)

def get_granules_list(granules):
    '''
    Function to get list of granules returned from maap.searchGranule()
    '''
    url_list = []
    output_list = []
    for res in granules:
        url_list.append(res.getDownloadUrl())

    for url in url_list:
        if url[0:5] == 's3://':
            url = url[5:].split('/')
            url[0] += '.s3.amazonaws.com'
            url = 'https://' + '/'.join(url)
        output_list.append(url)
    return output_list

def prep_filter_atl08_qual(atl08):
    '''
    Run this data prep on a df built from all CSVs from a DPS of extract_atl08.py for v003 of ATL08
    '''
    
    print("\nPre-filter data cleaning...")
     
    atl08.loc[( (atl08.orb_orient == 1 ) & (atl08['gt'].str.contains('r')) ), "beam_type"] = 'Strong' 
    atl08.loc[( (atl08.orb_orient == 1 ) & (atl08['gt'].str.contains('l')) ), "beam_type"] = 'Weak'
    atl08.loc[( (atl08.orb_orient == 0 ) & (atl08['gt'].str.contains('r')) ), "beam_type"] = 'Weak'
    atl08.loc[( (atl08.orb_orient == 0 ) & (atl08['gt'].str.contains('l')) ), "beam_type"] = 'Strong'
    print(f"\tGet beam type from orbit orientation and ground track: {atl08.beam_type.unique()}")


    cols_float = ['lat', 'lon', 'h_can', 'h_te_best', 'ter_slp']
    print("\tCast some columns to:")
    print(f"\t\ttype float: {cols_float}")
    atl08[cols_float] = atl08[cols_float].apply(pd.to_numeric, errors='coerce')

    cols_int = ['n_ca_ph', 'n_seg_ph', 'n_toc_ph']
    print(f"\t\ttype integer: {cols_int}")
    atl08[cols_int] = atl08[cols_int].apply(pd.to_numeric, downcast='signed', errors='coerce')
    
    atl08['y'] = pd.to_datetime(atl08['dt'].str.strip("b\'\"")).dt.year    
    atl08['m'] = pd.to_datetime(atl08['dt'].str.strip("b\'\"")).dt.month
    atl08['d'] = pd.to_datetime(atl08['dt'].str.strip("b\'\"")).dt.day
          
    return(atl08)

def get_topo_stack_fn(topo_stack_list_fn, in_tile_num, col_name='local_path'): # in earlier versions, 'local_path' was called 'location'
    # Find most recent topo stack path for tile in list of topo stack paths 
    all_topo_stacks_df = pd.read_csv(topo_stack_list_fn)
    stack_for_tile = all_topo_stacks_df[all_topo_stacks_df[col_name].str.contains("Copernicus_"+str(in_tile_num))]
    #[print(i) for i in stack_for_tile[col_name].to_list()]
    
    if len(stack_for_tile) > 0:
        topo_stack_fn = stack_for_tile[col_name].to_list()[0]
    else:
        topo_stack_fn = None
        
    return(topo_stack_fn)

def find_atl08_csv_tile(all_atl08_for_tile, all_atl08_csvs_df, seg_str, col_name='local_path', DEBUG=False): # in earlier versions, 'local_path' was called 'path'
    
    print("\tFind ATL08 CSVs you expect for a tile based on the h5 granule search...")
    # Change the small ATL08 H5 granule names to match the output filenames from extract_atl08.py (eg, ATL08_*_30m.csv)
    all_atl08_for_tile_CSVname = [os.path.basename(f).replace("ATL08", "ATL08"+seg_str).replace('.h5', seg_str+'.csv') for f in all_atl08_for_tile]

    if DEBUG:
        print(all_atl08_for_tile_CSVname)
    
    print('\t\t# of all ATL08 granules for tile: {}'.format(len(all_atl08_for_tile)))
    all_atl08_csvs = all_atl08_csvs_df[col_name].to_list()
    print('\t\t# of all_atl08_csvs: {}'.format(len(all_atl08_csvs)))
    
    # Get basenames of CSVs
    all_atl08_csvs_BASENAME = [os.path.basename(f) for f in all_atl08_csvs]
       
    # Get index of ATL08 in tile bounds from the large list of all ATL08 CSVs
    names_FOUND = [name for i, name in enumerate(all_atl08_for_tile_CSVname) if name in set(all_atl08_csvs_BASENAME)]
    names_NOT_FOUND = [name for i, name in enumerate(all_atl08_for_tile_CSVname) if name not in set(all_atl08_csvs_BASENAME)]
    
    if DEBUG:
        print("\t\tATL08 CSVs expected, but NOT FOUND: \n",names_NOT_FOUND)
        
    idx_FOUND = [all_atl08_csvs_BASENAME.index(name) for name in names_FOUND]
    
    if DEBUG:
        print("\t\tIndex of found: ", idx_FOUND)
        
    # DERP - cant get index of stuff that isnt there...
    ##idx_NOT_FOUND = [all_atl08_csvs_BASENAME.index(name) for name in names_NOT_FOUND]

    #print('\t\t# of matches between ATL08 CSVs and ATL08 granules for tile: {}'.format(len(idx_FOUND)))
    all_atl08_csvs_FOUND  = [all_atl08_csvs[i] for i in idx_FOUND]
    all_atl08_csvs_NOT_FOUND  = names_NOT_FOUND ##[all_atl08_csvs[i] for i in idx_NOT_FOUND]
    
    return(all_atl08_csvs_FOUND, all_atl08_csvs_NOT_FOUND)

def filter_atl08_bounds_tile_ept(in_ept_fn, in_tile_fn, in_tile_num, in_tile_layer, output_dir):
    '''Get bounds from a tile_id and apply to an EPT database
        Return a path to a GEOJSON that is a subset of the ATL08 db
    '''

    # Return the 4326 representation of the input <tile_id> geometry 
    tile_parts = CovariateUtils.get_index_tile(in_tile_fn, in_tile_num, buffer=0, layer = in_tile_layer)
    geom_4326 = tile_parts["geom_4326"]

    xmin, xmax = geom_4326[0:2]
    ymin, ymax = geom_4326[2:]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymax = transformer.transform(xmin, ymax)
    xmax, ymin = transformer.transform(xmax, ymin)
    pdal_tile_bounds = f"([{xmin}, {xmax}], [{ymin}, {ymax}])"

    # Spatial subset
    pipeline_def = [
        {
            "type": "readers.ept",
            "filename": in_ept_fn
        },
        {
            "type":"filters.crop",
            "bounds": pdal_tile_bounds
        },
        {
            "type" : "writers.text",
            "format": "geojson",
            "write_header": True
        }
    ]

    # Output the spatial subset as a geojson
    out_fn = os.path.join(output_dir, os.path.split(os.path.splitext(in_ept_fn)[0])[1] + "_" + in_tile_num + ".geojson")
    run_pipeline(pipeline_def, out_fn)

    return(out_fn)

def filter_atl08_bounds_clip(atl08_df, in_tile_geom_4326):
    
    atl08_gdf = gpd.GeoDataFrame(atl08_df, geometry=gpd.points_from_xy(atl08_df.lon, atl08_df.lat), crs='epsg:4326')
    atl08_gdf = gpd.clip(atl08_gdf, in_tile_geom_4326)
    
    return(atl08_gdf)

def filter_atl08_bounds(atl08_df=None, in_bounds=None, in_ept_fn=None, in_tile_fn=None, in_tile_num=None, in_tile_layer=None, output_dir=None, return_pdf=False):
    '''
    Filter an ATL08 database using bounds.
    Bounds can come from an input vector tile or a list: [xmin,xmax,ymin,ymax]
    '''
    out_fn = None
    print("\nFiltering by bounds: {}".format(in_bounds) )
    
    if all(v is not None for v in [in_ept_fn, in_tile_fn, in_tile_num, in_tile_layer, output_dir]):
        # TODO: finish, check, debug
        out_fn = filter_atl08_bounds_tile_ept(in_ept_fn, in_tile_fn, in_tile_num, in_tile_layer, output_dir)
        print("Filtering by bounds not yet ready for use with EPT." )
    elif in_bounds is not None and atl08_df is not None:
        print(f"\tBefore bounds filtering: \t\t{atl08_df.shape[0]} observations in the input dataframe.")
        print(f"\tMin lon: {atl08_df.lon.min()}")
        print(f"\tMax lon: {atl08_df.lon.max()}")
        xmin,xmax,ymin,ymax = in_bounds
        
        return_pdf = True
        print("\tData frame where:")
        print(f"\t\tlat>{ymin}")
        print(f"\t\tlat<{ymax}")
        print(f"\t\tlon>{xmin}")
        print(f"\t\tlon<{xmax}")
        if return_pdf :
            atl08_df = atl08_df[(atl08_df.lat > ymin) &
                                (atl08_df.lat < ymax) &
                                (atl08_df.lon > xmin) &
                                (atl08_df.lon < xmax)
                               ]
        print(f"After bounds filtering: \t\t{atl08_df.shape[0]} observations in the input dataframe.")
    else:
        print("Missing input args; can't filter. Check call.")
        os._exit(1)
    
    if return_pdf:
        if out_fn is not None:
            atl08_df = gpd.read(out_fn)
        return(atl08_df)
    else:
        print(out_fn)
        return(out_fn)

def filter_atl08_qual(input_fn=None, subset_cols_list=['rh25','rh50','rh60','rh70','rh75','rh80','rh85','rh90','rh95','h_can','h_max_can'], filt_cols =['h_can','h_dif_ref','m','msw_flg','beam_type','seg_snow'], thresh_h_can=None, thresh_h_dif=None, month_min=None, month_max=None, SUBSET_COLS=True, DO_PREP=True):
    '''
    Quality filtering Function
    Returns a data frame
    Note: beams 1 & 5 strong (better radiometric perf, sensitive), then beam 3 [NOT IMPLEMENTED]
    '''
    # TODO: filt col names: make sure you have these in the EPT db
    print("\nFiltering by quality")
    
    if not subset_cols_list:
        print("filter_atl08: Must supply a list of strings matching ATL08 column names returned from the input EPT")
        os._exit(1) 
    elif thresh_h_can is None:
        print("filter_atl08: Must supply a threshold for h_can")
        os._exit(1)    
    elif thresh_h_dif is None:
        print("filter_atl08: Must supply a threshold for h_dif_ref")
        os._exit(1)
    elif month_min is None or month_max is None:
        print("filter_atl08: Must supply a month_min and month_max")
        os._exit(1)  
        
    if input_fn is not None:
        if not isinstance(input_fn, pd.DataFrame):
            if input_fn.endswith('geojson'):
                atl08_df = gpd.read(input_fn)
            elif input_fn.endswith('csv'):
                atl08_df = pd.read_csv(input_fn)
            else:
                print("Input filename must be a CSV, GEOJSON, or pd.DataFrame")
                os._exit(1)
        else:
            atl08_df = input_fn
            
    if DO_PREP:
        # Run the prep to get fields needed (v003)
        atl08_df_prepd = prep_filter_atl08_qual(atl08_df)
    else:
        atl08_df_prepd = atl08_df
    
    
    # Check that you have the cols that are required for the filter
    filt_cols_not_in_df = [col for col in filt_cols if col not in atl08_df_prepd.columns] 
    if len(filt_cols_not_in_df) > 0:
        print("\tThese filter columns not found in input df: {}".format(filt_cols_not_in_df))
        print("\tNo quality filtering done. Returning original df.")
        return(atl08_df)
    
    atl08_df = None
    
    # Filtering
    #
    
    # Filter list (keep):
    #   h_ref_diff < thresh_h_dif
    #   h_can < thresh_h_can
    #   no LC forest masking: only forest LC classes no good b/c trees outside of forest aer of interest (woodlands, etc)
    #   msw = 0
    #   night better (but might exclude too much good summer data in the high northern lats)
    #   strong beam
    #   summer (june - mid sept)
    #   seg_snow == 'snow free land'
        
    #print("\nFiltering for quality:\n\tfor clear skies + strong beam + snow free land,\n\th_can < {},\n\televation diff from ref < {},\n\tmonths {}-{}".format(thresh_h_can, thresh_h_dif, month_min, month_max))
    print(f"\tBefore quality filtering: \t\t{atl08_df_prepd.shape[0]} observations in the input dataframe.")
    
    # Static quality filter flags for ABoVE AGB
    filt_params_static = [
                             ['msw_flg',0],
                             ['beam_type', 'Strong'],
                             ['seg_snow' , 1] #'snow free land'
                        ]

    atl08_df_filt = atl08_df_prepd
    for flag, val in filt_params_static:
        atl08_df_filt = atl08_df_filt[atl08_df_filt[flag] == val]
        print(f"\tAfter {flag}={val}: \t\t{atl08_df_filt.shape[0]} observations in the dataframe.")
    
    atl08_df_filt =  atl08_df_filt[
                                (atl08_df_filt.h_can < thresh_h_can) &
                                (atl08_df_filt.h_dif_ref < thresh_h_dif) &
                                (atl08_df_filt.m >= month_min ) & 
                                (atl08_df_filt.m <= month_max) 
                    ]    
    
    print(f"\tAfter all quality filtering: \t\t{atl08_df_filt.shape[0]} observations in the output dataframe.")
    
    atl08_df_prepd = None
    
    if SUBSET_COLS:
        subset_cols_list = ['lon','lat'] + subset_cols_list
        print("\tReturning a pandas data frame of filtered observations for columns: {}".format(subset_cols_list))
        print(f"\tData frame shape: {atl08_df_filt[subset_cols_list].shape} ")
        return(atl08_df_filt[subset_cols_list])
    else:
        print("\tReturning a pandas data frame of filtered observations for all columns")
        return(atl08_df_filt)