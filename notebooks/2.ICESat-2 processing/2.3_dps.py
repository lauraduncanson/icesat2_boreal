import pdal
import json
import os

import geopandas as gpd
from pyproj import CRS, Transformer

import argparse

from maap.maap import MAAP
maap = MAAP()

from CovariateUtils import get_index_tile #TODO: how to get this import right if its in a different dir
from FilterUtils import above_filter_atl08

def main():
    #
    # Access ATL08 obs in EPT, apply filter for quality, subset by bounds of a tile, subset by select cols, output as CSV
    #
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-ept", "--in_ept_fn", type=str, help="The input ept of ATL08 observations")
    parser.add_argument("-i", "--in_tile_fn", type=str, help="The input filename of a set of vector tiles that will define the bounds for ATL08 subset")
    parser.add_argument("-n", "--in_tile_num", type=int, help="The id number of an input vector tile that will define the bounds for ATL08 subset")
    parser.add_argument("-l", "--in_tile_layer", type=str, default=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-t_h_can", "--thresh_h_can", type=int, default=100, help="The threshold height below which ATL08 obs will be returned")
    parser.add_argument("-t_h_dif", "--thresh_h_dif", type=int, default=100, help="The threshold elev dif from ref below which ATL08 obs will be returned")
    parser.add_argument("-m_min", "--month_min", type=int, default=6, help="The min month of each year for which ATL08 obs will be used")
    parser.add_argument("-m_max", "--month_max", type=int, default=9, help="The max month of each year for which ATL08 obs will be used")
    parser.add_argument('-ocl', '--out_cols_list', nargs='+', default=[], help="A select list of strings matching ATL08 col names from the input EPT that will be returned in a pandas df after filtering and subsetting")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The output dir of the filtered and subset ATL08 csv")

    args = parser.parse_args()
    if args.in_ept_fn == None:
        print("Input a filename of the EPT database of ATL08 obs tiles that will be quality-filtered and subset by tile")
        os._exit(1)
    if args.in_tile_fn == None:
        print("Input a filename of the vector tiles that represents the arrangement by which the ATL08 obs will be organized")
        os._exit(1)
    elif args.in_tile_num == None:
        print("Input a specific tile id from the vector tiles the organize the ATL08 obs")
        os._exit(1)
    elif args.in_tile_layer == None:
        print("Input a layer name from the tile vector file")
        os._exit(1)   
        
    in_tile_fn = args.in_tile_fn
    in_tile_num = args.in_tile_num
    in_tile_layer = args.in_tile_layer
    thresh_h_can = args.thresh_h_can
    thresh_h_dif = args.thresh_h_dif
    month_min = args.month_min
    month_max = args.month_max
    out_cols_list = args.out_cols_list
    output_dir = args.output_dir
    
    # Return the 4326 representation of the input <tile_id> geometry 
    tile_parts = get_index_tile(in_tile_fn, in_tile_num, buffer=0, layer = in_tile_layer)
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
    
    # Filter based on a standard filter_atl08() function that we use across all notebooks, scripts, etc
    atl08_pdf_filt = above_filter_atl08(input_geojson, out_cols_list)

    # CSV the file
    cur_time = time.strftime("%Y%m%d%H%M%S")
    out_csv_fn = os.path.join(output_dir, "atl08_filt_"+cur_time+".csv")
    atl08_pdf_filt.to_pickle(out_csv_fn,index=False, encoding="utf-8-sig")
    print("Wrote output csv: ", out_csv_fn)

if __name__ == "__main__":
    main()