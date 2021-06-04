import pdal
import json
import os

import pandas as pd
import geopandas as gpd

from pyproj import CRS, Transformer

import CovariateUtils #< - DO THIS IMPORT CORRECTLY

def filter_atl08_ept(in_ept_fn, in_tile_fn, in_tile_num, in_tile_layer, output_dir, return_pdf=False):
    
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
    
    if(return_pdf):
        atl08_df = gpd.read(out_fn)
        return(atl08_df)
    else:
        print(out_fn)
        return(out_fn)

def filter_atl08(input_fn=None, out_cols_list=['rh25','rh50','rh60','rh70','rh75','rh80','rh85','rh90','rh95','h_can','h_can_max'], thresh_h_can=None, thresh_h_dif=None, month_min=None, month_max=None):
    
    if not out_cols_list:
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
        if input_fn.endswith('geojson'):
            atl08_df = gpd.read(input_fn)
        elif input_fn.endswith('csv'):
            atl08_df = pd.read_csv(input_fn)
        elif isinstance(input_fn, pd.DataFrame):
            atl08_df = input_fn
        else:
            print("Input filename must be a CSV, GEOJSON, or pd.DataFrame")
            os._exit(1)
                
    # Filtering
    #
    # Filter list (keep):
    #   h_ref_diff < THRESH_h_dif
    #   h_can < THRESH_h_can
    #   no LC forest masking: only forest LC classes no good b/c trees outside of forest aer of interest (woodlands, etc)
    #   msw = 0
    #   night better (but might exclude too much good summer data in the high northern lats)
    #   strong beam
    #   summer (june - mid sept)
    #   seg_snow == 'snow free land'
        
    print("\nFiltering for quality:\n\tfor clear skies + strong beam + snow free land,\n\th_can < {},\n\televation diff from ref < {},\n\tmonths {}-{}".format(thresh_h_can, thresh_h_dif, month_min, month_max))
    atl08_df_filt =  atl08_df[
                                (atl08_df.h_can < THRESH_h_can) &
                                (atl08_df.h_dif_ref < THRESH_h_dif) &
                                (atl08_df.m >= month_min ) & (atl08_pdf.m <= month_max) &
                                # Hard coded quality flags for ABoVE AGB
                                (atl08_df.msw_flg == 0) &
                                #(atl08_df.night_flg == 'night') & # might exclude too much good summer data in the high northern lats
                                (atl08_df.beam_type == 'Strong') & 
                                (atl08_df.seg_snow == 'snow free land')
                    ]
    
    out_cols_list = ['lon','lat', out_cols_list]
    
    print(f"Before filtering: {atl08_df.shape[0]} observations in the input dataframe.")
    print(f"After filtering: {atl08_df_filt.shape[0]} observations in the output dataframe.")
    print("Returning a pandas data frame of filtered observations for columns: {}".format(out_cols_list))
    
    return(atl08_df_filt[out_cols_list])