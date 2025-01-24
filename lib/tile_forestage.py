import os, sys
import h5netcdf
import xarray as xr
import fsspec

import argparse

import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np

# Clip and plot
def clip_to_gdf(ds, gdf, NODATA_VAL):
    
    # Assign spatial reference and enable rioxarray for geospatial operations
    ds = ds.rio.write_crs("EPSG:4326") 
    
    print(f'Clipping to vector...')
    # Clip the dataset to the polygon and handle nodata
    clipped_ds = ds.rio.clip([mapping(gdf.to_crs(4326).iloc[0].geometry)], from_disk=True)
    
    if False:
        print(f'Setting nodata value to {NODATA_VAL}...')
        # Set nodata value for pixels outside the clipped area
        clipped_ds = clipped_ds.rio.write_nodata(NODATA_VAL) # not working as expected set nodata inside clipped area correctly to -9999 - but outside goes to 0.0
    
    #print(f'Handling data values: min = {MIN_VAL}, nodata = {NODATA_VAL}...')
    #clipped_ds = clipped_ds.where(clipped_ds > MIN_VAL, clipped_ds, NODATA_VAL) # not working as expected

    return clipped_ds

def process_ds_all(ds, tile_bounds, gdf, TIME_IDX=1, VAR_NAME="forest_age"):
    
    print(f'Performing time selection and general geo-slice for {VAR_NAME} with bound {tile_bounds}...')
    ds_sub = ds[VAR_NAME].isel(time=TIME_IDX).sel(longitude = slice(tile_bounds[0]-1, tile_bounds[2]+1), 
                                                              latitude =  slice(tile_bounds[3]+1, tile_bounds[1]-1))

    clipped_ds = clip_to_tile(ds_sub, gdf)
    
    return clipped_ds

def tile_forestage_(TILE_NUM, 
                      URL,
                      VECTOR_FN,
                      ID_COL,
                      VAR_NAME = "forest_age",
                      MIN_VAL = 1.0,
                      NODATA_VAL = np.nan,
                      YEAR = '2020',
                      OUTDIR = '/projects/my-public-bucket/local_output/forest_age',
                      RETURN_STACK=False):
    
    '''Given an ensemble forestage dataset (the dim called 'members') and vector , clip to a tile, calc mean and std layers, stack, and output a geotiff 
    '''
    
    gdf = gpd.read_file(VECTOR_FN)
    gdf_sub = gdf[gdf[ID_COL] == TILE_NUM]
    tile_bounds = [round(d, 0) for d in gdf_sub.to_crs(4326).total_bounds]

    TIME_IDX = 0
    if YEAR == '2020': TIME_IDX = 1

    fs = fsspec.filesystem('http')

    with xr.open_dataset(fs.open(URL)) as ds:
        
        print(f'Opened the data: {URL}')
        #clipped_ds = process_ds_all(ds, tile_bounds, gdf_sub, TIME_IDX=TIME_IDX, VAR_NAME=VAR_NAME)
        print(f'Performing time selection and (for mem efficiency?) a general (bbox) geo-slice for {VAR_NAME} with bound {tile_bounds}...')
        ds_sub = ds[VAR_NAME].isel(time=TIME_IDX).sel(longitude = slice(tile_bounds[0]-1, tile_bounds[2]+1), latitude =  slice(tile_bounds[3]+1, tile_bounds[1]-1))
        # Clip
        clipped_ds = clip_to_gdf(ds_sub, gdf_sub,
                                 #MIN_VAL, 
                                 NODATA_VAL)
        
        print(f'Calcing ensemble mean and std...')
        clipped_array = xr.concat([clipped_ds.mean(dim='members'), clipped_ds.std(dim='members')], 
                              dim = 'time', compat="no_conflicts").rio.write_crs("EPSG:4326") 
        if True:
            # Turn this DataArray into a Dataset so you can add bandnames
            clipped_dataset = xr.Dataset()
            bandnames_list = ['mean_forest_age_ensemble','std_forest_age_ensemble']
            
            for i, band in enumerate(bandnames_list):
                clipped_dataset[band] = clipped_array[i]
                
                # Set nodata here
                print(f'Setting data of band {band} that is less than {MIN_VAL} to nodata value of {NODATA_VAL}...')
                clipped_dataset[band] = clipped_dataset[band].where(clipped_dataset[band]>=MIN_VAL, other=NODATA_VAL)
                
            clipped_array = clipped_dataset
            clipped_dataset = None
        
        # write stack (an xarray dataset) as a multi-band tiff
        clipped_fn = os.path.join(OUTDIR, f"{VAR_NAME.replace('_','')}_{YEAR}_{int(TILE_NUM):07}.tif")
        
        print(f'Writing the geotiff: {clipped_fn}...')
        clipped_array.rio.to_raster(clipped_fn, tiled=True)
        print(f'Finished writing.')
        
        if RETURN_STACK:
            return clipped_array
        else:
            return clipped_fn

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_url", type=str, default='https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/GAMIv2-1_2010-2020_100m.nc', help="The input of the forest age")
    parser.add_argument("-v", "--in_vector_fn", type=str, default='/projects/shared-buckets/montesano/databank/boreal_tiles_v004.gpkg')
    parser.add_argument("-n", "--in_id_num", type=int, help="The id number of an input vector tile that will define the bounds for stack creation")
    parser.add_argument("-y", "--year", type=str, default='2020', help="The year of the forest age data (2010, 2020)")
    parser.add_argument("--in_id_col", type=str, default="tile_num", help="The column of the tile layer name of the stack tiles dataset that holds the tile num")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The path for the output stack")
    parser.add_argument("-tmp", "--tmp_out_path", type=str, default="/tmp", help="The tmp path for the output stack")
    parser.add_argument('--return_stack', dest='return_stack', action='store_true', help='Return a 3d numpy masked array.')
    parser.set_defaults(return_stack=False)

    args = parser.parse_args()
    
    if args.in_url == None:
        print("Input a url for the forest age data")
        os._exit(1)
    elif args.in_id_num == None:
        print("Input a specific id from the vector tiles the organize the stacks")
        os._exit(1)
    if args.output_dir == None:
        print("Output dir set to {}".format(args.tmp_out_path))
        output_dir = args.tmp_out_path
    else:
        output_dir = args.output_dir
    
    print("\n---Running tile_forestage_()---\n")
    tile_forestage_(
        TILE_NUM = args.in_id_num,
        URL = args.in_url,
        VECTOR_FN = args.in_vector_fn,
        ID_COL = args.in_id_col,
        YEAR = args.year,
        OUTDIR = output_dir
               )

if __name__ == "__main__":
    main()