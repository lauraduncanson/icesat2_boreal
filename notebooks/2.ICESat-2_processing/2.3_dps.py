import pdal
import json
import os
import glob

import geopandas as gpd
from pyproj import CRS, Transformer

import argparse

from maap.maap import MAAP
maap = MAAP()

import sys
sys.path.append("/projects/icesat2_boreal/notebooks/3.Gridded_product_development/")
sys.path.append('/projects/code/icesat2_boreal/notebooks/3.Gridded_product_development')

#TODO: how to get this import right if its in a different dir
import CovariateUtils 
import FilterUtils
import ExtractUtils

#TODO: do this right also
#import 3.1.5_dps
#import 3.1.2_dps
# https://stackoverflow.com/questions/1828127/how-to-reference-python-package-when-filename-contains-a-period
import imp
with open('3.1.5_dps.py', 'rb') as script:
    models_admin = imp.load_module('do_3_1_5_dp', script, '3.1.5_dps.py',('.py', 'rb', imp.PY_SOURCE))
with open('3.1.2_dps.py', 'rb') as script:
    models_admin = imp.load_module('do_3_1_2_dp', script, '3.1.2_dps.py',('.py', 'rb', imp.PY_SOURCE))



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
    parser.add_argument("--local", dest='local', action='store_true', help="Dictate whether landsat covars is a run using local paths")
    parser.set_defaults(local=False)
    parser.add_argument("-t_h_can", "--thresh_h_can", type=int, default=100, help="The threshold height below which ATL08 obs will be returned")
    parser.add_argument("-t_h_dif", "--thresh_h_dif", type=int, default=100, help="The threshold elev dif from ref below which ATL08 obs will be returned")
    parser.add_argument("-m_min", "--month_min", type=int, default=6, help="The min month of each year for which ATL08 obs will be used")
    parser.add_argument("-m_max", "--month_max", type=int, default=9, help="The max month of each year for which ATL08 obs will be used")
    parser.add_argument('-ocl', '--out_cols_list', nargs='+', default=[], help="A select list of strings matching ATL08 col names from the input EPT that will be returned in a pandas df after filtering and subsetting")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The output dir of the filtered and subset ATL08 csv")
    parser.add_argument("-dps_dir", "--dps_output_dir", type=str, default=None, help="The top-level DPS output dir for the ATL08 csv files")
    parser.add_argument("-date_start", type=str, default="06-01", help="Seasonal start MM-DD")
    parser.add_argument("-date_end", type=str, default="09-30", help="Seasonal end MM-DD")
    parser.add_argument('--maap-query', dest='maap-query', action='store_true', help='Run a MAAP query by tile to return list of ATL08 h5 that forms the database of ATL08 observations')
    parser.set_defaults(maap-query=False)
    parser.add_argument('--do_30m', dest='do_30m', action='store_true', help='Turn on 30m ATL08 extraction')
    parser.set_defaults(do_30m=False)

    args = parser.parse_args()
    if args.in_ept_fn == None and not args.maap-query:
        print("The flag 'maap-query' is false so you need an input filename of the EPT database of ATL08 obs tiles that will be quality-filtered and subset by tile")
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
    do_30m = args.do_30m
    dps_dir = args.dps_dir
    
    # TODO: make this an arg
    years_list = [2019, 2020, 2021]
    
    seg_str = '_100m'
    if do_30m:
        seg_str = '_30m'
    
    if args.maap_query and dps_dir is not None:
        
        print("\nDoing MAAP query by tile bounds to find all intersecting ATL08 ")
        # Get a list of all ATL08 H5 granule names intersecting the tile (this will be a small list)
        all_atl08_for_tile = ExtractUtils.get_h5_list(tile_num=in_tile_num, tile_fn=in_tile_fn, layer=in_tile_layer, DATE_START=args.date_start, DATE_END=args.date_end, YEARS=years_list)
        
        # Change the small ATL08 H5 granule names to match the output filenames from extract_atl08.py (eg, ATL08_*_30m.csv)
        all_atl08_csvs_for_tile_BASENAME = [os.path.basename(f).replace('.h5', seg_str+'.csv') for f in all_atl08_for_tile]
        
        # Get a list of all ATL08 CSV files from (from extract_atl08) (this will be a large boreal list)
        all_atl08_csvs = glob.glob(dps_dir + "/**/ATL08*" + seg_str + ".csv", recursive=True)
        all_atl08_csvs_BASENAME = [os.path.basename(f) for f in all_atl08_csvs]
        
        # Get index of ATL08 in tile bounds from the large list of all ATL08 CSVs
        idx = [i for i, e in enumerate(all_atl08_csvs_for_tile_BASENAME) if e in set(all_atl08_csvs_BASENAME)]
        # Get the subset of all ATL08 CSVs that just correspond to the ATL08 H5 intersecting the current tile
        all_atl08_csvs_for_tile = [all_atl08_for_tile[x] for x in idx]       
        
        # Check to make sure these are in fact files (necessary?)
        all_atl08_csvs_NOT_FOUND = []
        for file in all_atl08_csvs_for_tile: 
            if not os.path.isfile(file):
                all_atl08_csvs_NOT_FOUND.append(file)

        all_atl08_csvs_FOUND = [x for x in all_atl08_csvs_for_tile if x not in all_atl08_csvs_NOT_FOUND]
        
        # Merge all ATL08 CSV files for the current tile into a pandas df
        print("Creating pandas data frame...")
        atl08 = pd.concat([pd.read_csv(f) for f in all_atl08_csvs_FOUND ], sort=False)
        
        print("\nFiltering by tile: {}".format(in_tile_num))
    
        # Get tile bounds as xmin,xmax,ymin,ymax
        in_bounds = FilterUtils.reorder_4326_bounds(in_tile_fn, in_tile_num, buffer=0, layer=in_tile_layer)
        
        # Now filter ATL08 obs by tile bounds
        atl08 = FilterUtils.filter_atl08_bounds(atl08_df=atl08, in_bounds=in_bounds)
        
    elif args.maap_query and dps_dir is None:
        print("\nNo DPS dir specified: cant get ATL08 CSV list to match with tile bound results from MAAP query.\n")
        os._exit(1)
    else:
        # Filter by bounds: EPT with a the bounds from an input tile
        atl08 = FilterUtils.filter_atl08_bounds_tile_ept(in_ept_fn, in_tile_fn, in_tile_num, in_tile_layer, output_dir, return_pdf=True)
    
    ## Filter by quality: based on a standard filter_atl08_qual() function that we use across all notebooks, scripts, etc
    #atl08_pdf_filt = FilterUtils.filter_atl08_qual(atl08, out_cols_list)
    # Filter by quality
    atl08_pdf_filt = FilterUtils.filter_atl08_qual(atl08, SUBSET_COLS=True, 
                                                       subset_cols_list=['rh25','rh50','rh60','rh70','rh75','rh80','rh85','rh90','rh95','h_can','h_max_can'], 
                                                       filt_cols=['h_can','h_dif_ref','m','msw_flg','beam_type','seg_snow', 'seg_landcov'], 
                                                       thresh_h_can=100, thresh_h_dif=100, month_min=6, month_max=9)
    atl08=None
    
    # Convert to geopandas data frame in lat/lon
    atl08_gdf = GeoDataFrame(atl08_pdf_filt, geometry=gpd.points_from_xy(atl08_pdf_filt.lon, atl08_pdf_filt.lat), crs='epsg:4326')
    out_name_stem = "atl08_filt"
    atl08_pdf_filt=None
    
    ### Below here should be re-worked to follow final chunk of nb 2.3 (6/15/2021)
    #
    if False:
        # Extract topo covar values to ATL08 obs (doing a reproject to tile crs)
        # TODO: consider just running 3.1.5_dpy.py here to produce this topo stack right before extracting its values
        topo_covar_fn = do_3_1_5_dp.main(in_tile_fn=in_tile_fn, in_tile_num=in_tile_num, tile_buffer_m=120, in_tile_layer=in_tile_layer, topo_tile_fn='https://maap-ops-dataset.s3.amazonaws.com/maap-users/alexdevseed/dem30m_tiles.geojson')
        atl08_gdf_out = ExtractUtils.extract_value_gdf(topo_covar_fn, atl08_gdf, ["elevation","slope","tsri","tpi", "slopemask"], reproject=True)
        out_name_stem = out_name_stem + "_topo"
    
    if False:
        # Extract landsat covar values to ATL08 obs
        # TODO: consider just running 3.1.2_dpy.py here
        landsat_covar_fn = do_3_1_2_dps.main(in_tile_fn=in_tile_fn, in_tile_num=in_tile_num, in_tile_layer=in_tile_layer, sat_api='https://landsatlook.usgs.gov/sat-api', local=args.local)
        atl08_gdf_out = ExtractUtils.extract_value_gdf(landsat_covar_fn, atl08_gdf_out, ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'ValidMask', 'Xgeo', 'Ygeo'], reproject=False)
        out_name_stem = out_name_stem + "_landsat"
        
    # CSV the file
    cur_time = time.strftime("%Y%m%d%H%M%S")
    out_csv_fn = os.path.join(output_dir, out_name_stem + "_" + cur_time + ".csv")
    atl08_gdf_out.to_csv(out_csv_fn,index=False, encoding="utf-8-sig")
    
    print("Wrote output csv of filtered ATL08 obs with topo and Landsat covariates for tile {}: {}".format(in_tile_num, out_csv_fn) )

if __name__ == "__main__":
    main()