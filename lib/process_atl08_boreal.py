import sys, os
import pandas as pd
import geopandas as gpd
import logging 
import sliderule
from sliderule import icesat2

import FilterUtils
import CovariateUtils
import mosaiclib
from mosaiclib import *

from pathlib import Path
import glob
import math

def gdf_to_sliderulepoly(gdf):
    '''
    Return a dictionary of from a geodataframe of polygon coordinates needed for sliderule
    works for any polygon
    '''
    dict_list = []
    for i, lat in enumerate(gdf.get_coordinates()['y']):
        d = {'lat':lat, 'lon':gdf.get_coordinates().iloc[i]['x']}
        dict_list.append(d)
        if len(dict_list) == len(gdf.get_coordinates())-1: break

    return dict_list

def truncate(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def process_atl08_boreal(polygon_id, polygon_gdf_fn, id_col_num = 'tile_num', t0_year=2020, t1_year=2020, minmonth=6, maxmonth=9, seg_length = 30, 
                         outdir='/projects/my-private-bucket/data/process_atl08_boreal',
                         atl08_cols_list=['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can','h_max_can', 'ter_slp','h_te_best', 'seg_landcov','sol_el','y','m','doy'],
                         RETURN_DF=False, DEBUG=False):
    
    '''Runs sliderule's implementation of PhoReal to process a clipped and filtered geodataframe of custom ATL08 along-track segments designed for boreal forests
        + finds ATL03 for a polygon_id from geodataframe at path gdf_fn
        + filters by year and month
        + calculates custom ATL08 observations representing along-track segents of length 'seg_length' using sliderule
        + applies quality filtering customized for boreal forests
    '''
    
    # Subset polygon_gdf by tile and reproject to WGS84
    polygon_gdf = gpd.read_file(polygon_gdf_fn)
    polygon_gdf = polygon_gdf[polygon_gdf[id_col_num] == polygon_id].to_crs(4326)
    
    # Set SlideRule parameters
    maxday = 30
    if maxmonth in [2]: maxday = 29
    if maxmonth in [1,3,5,7,8,10,12]: maxday = 31
    
    #outdir = os.path.join(outdir,f'{seg_length:03}m')
    out_name = os.path.join(outdir, f'atl08_006_{seg_length:03}m_{t0_year}_{t1_year}_{minmonth:02}_{maxmonth:02}')
    
    # Get list of points dictionaries that have 'lat' and 'lon'
    points_dict_list = gdf_to_sliderulepoly(polygon_gdf)
    
    # Handle antimeridian 5 decimal issue
    for i,point in enumerate(points_dict_list):
        if abs(point['lon']) > 179.9999:
            orig_lon = point['lon']
            truncated_lon = truncate(point['lon'], 4)
            points_dict_list[i]['lon'] = truncated_lon
            print(f"Truncated {orig_lon} t0 {truncated_lon}")

    params_atl08 = {
            #"output": { "path": f"{out_name}_{polygon_id:04}.parquet", "format": "parquet", "open_on_complete": True }, # open on compelte not working as expected?
            "poly": points_dict_list,
            "t0": f'{t0_year}-{minmonth:02}-01T00:00:00Z',
            "t1": f'{t1_year}-{maxmonth:02}-{maxday}T00:00:00Z',
            "srt": icesat2.SRT_LAND,
            "len": seg_length,
            "res": seg_length,
            "pass_invalid": True, 
            "atl08_class": ["atl08_ground", "atl08_canopy", "atl08_top_of_canopy"],
            "atl08_fields": ["canopy/h_canopy_uncertainty","h_dif_ref","msw_flag","sigma_topo","segment_landcover","canopy/segment_cover","segment_snowcover","terrain/h_te_uncertainty"], #'segment_cover' and 'h_canopy_uncertainty' need to be added
            #"atl08_fields": ["h_dif_ref","msw_flag","sigma_topo","segment_landcover"],
            "phoreal": {"binsize": 1.0, "geoloc": "center", "above_classifier": True, "use_abs_h": False, "send_waveform": False}
        }
    if DEBUG:
        print(f"Polygon:\n{gdf_to_sliderulepoly(polygon_gdf)}")
    ###############################
    # Run SlideRule's implementation of PhoReal processing of ATL03 into custom ATL08
    # https://slideruleearth.io/web/rtd/api_reference/icesat2.html#atl08p
    atl08 = icesat2.atl08p(params_atl08, keep_id=True)
    print(f'ATL08 obs for tile {polygon_id:06} from sliderule: {atl08.shape[0]}')
    
    ###############################
    #### Project-specific formatting
    # Format for filtering
    # Unpack the canopy_h_metrics into discrete fields
    atl08['rh25'] = [l[3] for l in atl08['canopy_h_metrics']]
    atl08['rh50'] = [l[8] for l in atl08['canopy_h_metrics']]
    atl08['rh60'] = [l[10] for l in atl08['canopy_h_metrics']]
    atl08['rh70'] = [l[12] for l in atl08['canopy_h_metrics']]
    atl08['rh75'] = [l[13] for l in atl08['canopy_h_metrics']]
    atl08['rh80'] = [l[14] for l in atl08['canopy_h_metrics']]
    atl08['rh85'] = [l[15] for l in atl08['canopy_h_metrics']]
    atl08['rh90'] = [l[16] for l in atl08['canopy_h_metrics']]
    atl08['rh95'] = [l[17] for l in atl08['canopy_h_metrics']]

    # Rename fields with 'canopy'
    atl08['h_canopy_uncertainty'] = atl08['canopy/h_canopy_uncertainty']
    atl08['segment_cover'] = atl08['canopy/segment_cover']
    atl08['h_te_uncertainty'] = atl08['terrain/h_te_uncertainty']

    # Drop 'waveform' and 'canopy_h_metrics' fields 
    atl08 = atl08.drop(columns=[#'waveform', 
                                'canopy_h_metrics','canopy/h_canopy_uncertainty','canopy/segment_cover','terrain/h_te_uncertainty'])
    
    # Time is the index field
    atl08.reset_index(inplace=True)
    atl08['y'] = atl08.time.dt.year
    atl08['m'] = atl08.time.dt.month
    atl08['d'] = atl08.time.dt.day
    atl08['doy'] = atl08.time.dt.dayofyear
    atl08.drop('time', axis=1, inplace=True) # folium mapping a gdf with a datetime field doesnt work
    
    ###############################
    # Quality filtering designed for boreal forest
    atl08, atl08_meta = FilterUtils.filter_atl08_qual_v5(atl08, atl08_cols_list = atl08_cols_list, RETURN_METADATA=True)
    
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out_atl08_filt_fn = f'{out_name}_filt_{polygon_id:06}.parquet' ### updated from 05 to 06 padding
    atl08.to_parquet(out_atl08_filt_fn)
    print(f'File written:\t{out_atl08_filt_fn}')
    
    # Write filtering metadata
    atl08_meta['tile_num'] = polygon_id
    atl08_meta.to_csv(f'{out_name}_filt_metadata_{polygon_id:06}.csv')
    
    if RETURN_DF:
        return atl08
    else:
        atl08 = None