import pdal
import json
import os

import geopandas as gpd
from pyproj import CRS, Transformer

import argparse

from maap.maap import MAAP
maap = MAAP()

#TODO: how to get this import right if its in a different dir
import CovariateUtils 
import FilterUtils
import ExtractUtils

#TODO: do this right also
import 3.1.5_dps
import 3.1.2_dps


def main():
    #
    # Access ATL08 obs in EPT, apply filter for quality, subset by bounds of a tile, subset by select cols, extract values of covars, output as CSV
    #
    
    #TODO: how to specify (where will these data sit by default - in original dps_output dubdir, or pulled to the top level?):
    #     topo_covar_fn, landsat_covar_fn
    # Solution: just call 3.1.5_dpy and 3.1.2_dps and return the stack_fn from each call?
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-ept", "--in_ept_fn", type=str, help="The input ept of ATL08 observations")
    parser.add_argument("-i", "--in_tile_fn", type=str, help="The input filename of a set of vector tiles that will define the bounds for ATL08 subset")
    parser.add_argument("-n", "--in_tile_num", type=int, help="The id number of an input vector tile that will define the bounds for ATL08 subset")
    parser.add_argument("-lyr", "--in_tile_layer", type=str, default=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-l", "--local", type=bool, default=False, help="Dictate whether landsat covars is a run using local paths")
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
      
    in_ept_fn = args.in_ept_fn
    in_tile_fn = args.in_tile_fn
    in_tile_num = args.in_tile_num
    in_tile_layer = args.in_tile_layer
    thresh_h_can = args.thresh_h_can
    thresh_h_dif = args.thresh_h_dif
    month_min = args.month_min
    month_max = args.month_max
    out_cols_list = args.out_cols_list
    output_dir = args.output_dir
    
    # Filter by bounds: EPT with a the bounds from an input tile
    atl08_fn = FilterUtils.filter_atl08_bounds_tile_ept(in_ept_fn, in_tile_fn, in_tile_num, in_tile_layer, output_dir, return_pdf=False)
    
    # Filter by quality: based on a standard filter_atl08_qual() function that we use across all notebooks, scripts, etc
    atl08_pdf_filt = FilterUtils.filter_atl08_qual(atl08_fn, out_cols_list)
    
    # Convert to geopandas data frame in lat/lon
    atl08_gdf = GeoDataFrame(atl08_pdf_filt, geometry=gpd.points_from_xy(atl08_pdf_filt.lon, atl08_pdf_filt.lat), crs='epsg:4326')
    
    # Extract topo covar values to ATL08 obs (doing a reproject to tile crs)
    # TODO: consider just running 3.1.5_dpy.py here to produce this topo stack right before extracting its values
    topo_covar_fn = 3.1.5_dps.main(in_tile_fn=in_tile_fn, in_tile_num=in_tile_num, tile_buffer_m=120, in_tile_layer=in_tile_layer, topo_tile_fn='https://maap-ops-dataset.s3.amazonaws.com/maap-users/alexdevseed/dem30m_tiles.geojson')
    atl08_gdf_topo = ExtractUtils.extract_value_gdf(topo_covar_fn, atl08_gdf, ["elevation","slope","tsri","tpi", "slopemask"], reproject=True)
    
    # Extract landsat covar values to ATL08 obs
    # TODO: consider just running 3.1.2_dpy.py here
    landsat_covar_fn = 3.1.2_dps.main(in_tile_fn=in_tile_fn, in_tile_num=in_tile_num, in_tile_layer=in_tile_layer, sat_api='https://landsatlook.usgs.gov/sat-api', local=args.local)
    atl08_gdf_topo_landsat = ExtractUtils.extract_value_gdf(<<landsat_covar_fn>>, atl08_gdf_topo, ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'ValidMask', 'Xgeo', 'Ygeo'], reproject=False):

    # CSV the file
    cur_time = time.strftime("%Y%m%d%H%M%S")
    out_csv_fn = os.path.join(output_dir, "atl08_filt_topo_landsat_"+cur_time+".csv")
    atl08_gdf_topo_landsat.to_csv(out_csv_fn,index=False, encoding="utf-8-sig")
    
    print("Wrote output csv of filtered ATL08 obs with topo and Landsat covariates for tile {}: {}".format(in_tile_num, out_csv_fn) )

if __name__ == "__main__":
    main()