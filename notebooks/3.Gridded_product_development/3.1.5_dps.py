#! /usr/bin/env python

import os
import boto3
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.session import AWSSession 
from rasterio.warp import * #TODO: limit to specific needed modules
from rasterio.merge import merge
from rasterio.crs import CRS
import argparse
from CovariateUtils import write_cog, get_index_tile
from CovariateUtils_topo import *

def main():
    '''Command line script to create topo stacks by vector tile id.
    example cmd line call: python 3.1.5_dps.py --in_tile_fn '/projects/maap-users/alexdevseed/boreal_tiles.gpkg' --in_tile_num 18822 --tile_buffer_m 120 --in_tile_layer "boreal_tiles_albers" -o '/projects/tmp/Topo/'
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_tile_fn", type=str, help="The input filename of a set of vector tiles that will define the bounds for stack creation")
    parser.add_argument("-n", "--in_tile_num", type=int, help="The id number of an input vector tile that will define the bounds for stack creation")
    parser.add_argument("-b", "--tile_buffer_m", type=int, default=None, help="The buffer size (m) applied to the extent of the specified stack tile")
    parser.add_argument("-l", "--in_tile_layer", type=str, default=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The path for the output stack")
    parser.add_argument("-r", "--res", type=int, default=30, help="The output resolution of the stack")
    parser.add_argument("-topo", "--topo_tile_fn", type=str, default="/projects/maap-users/alexdevseed/dem30m_tiles.geojson", help="The filename of the topo's set of vector tiles")
    parser.add_argument("-tmp", "--tmp_out_path", type=str, default="/projects/tmp", help="The tmp out path for the clipped topo cog before topo calcs")
    parser.add_argument("-tsrc", "--topo_src_name", type=str, default="Copernicus", help="Name to identify the general source of the topography")

    args = parser.parse_args()
    
    if args.in_tile_fn == None:
        print("Input a filename of the vector tiles that represents the arrangement by which the output stacks will be organized")
        os._exit(1)
    elif args.in_tile_num == None:
        print("Input a specific tile id from the vector tiles the organize the stacks")
        os._exit(1)
    elif args.tile_buffer_m == None:
        print("Input a tile buffer distance in meters")
        os._exit(1)
    elif args.in_tile_layer == None:
        print("Input a layer name from the stack tile vector file")
        os._exit(1)
    
    if args.output_dir == None:
        print("Output dir set to {}".format(args.tmp_out_path))
        output_dir = args.tmp_out_path
    else:
        output_dir = args.output_dir
        

    stack_tile_fn = args.in_tile_fn
    stack_tile_id = args.in_tile_num
    stack_tile_layer = args.in_tile_layer
    res = args.res
    tile_buffer_m = args.tile_buffer_m
    topo_tile_fn = args.topo_tile_fn
    tmp_out_path = args.tmp_out_path
    topo_src_name = args.topo_src_name
    
    # Return the 4326 representation of the input <tile_id> geometry that is buffered in meters with <tile_buffer_m>
    tile_parts = get_index_tile(stack_tile_fn, stack_tile_id, buffer=tile_buffer_m, layer = stack_tile_layer)
    geom_4326_buffered = tile_parts["geom_4326_buffered"]
    
    # Read the topography index file
    dem_tiles = gpd.read_file(topo_tile_fn)

    # intersect with the bbox tile
    dem_tiles_selection = dem_tiles.loc[dem_tiles.intersects(geom_4326_buffered.iloc[0])]

    # Set up and aws session
    aws_session = AWSSession(boto3.Session())
    
    # Get the s3 urls to the granules
    file_s3 = dem_tiles_selection["s3"].to_list()
    file_s3.sort()
    print("The DEM filename(s) intersecting the {} m bbox for tile id {}:\n".format(str(tile_buffer_m), str(stack_tile_id)), '\n'.join(file_s3))
    
    # Create a mosaic from all the images
    with rio.Env(aws_session):
        sources = [rio.open(raster, 'r') for raster in file_s3]    

    # Merge the source files
    print("Bounds:\n", *tile_parts['bbox_4326_buffered'])
    mosaic, out_trans = merge(sources, bounds = tile_parts['bbox_4326_buffered'])
    
    #
    # Writing tmp elevation COG so that we can read it in the way we need to (as a gdal.Dataset)
    #
    if (not os.path.isdir(tmp_out_path)): os.mkdir(tmp_out_path)
    tileid = '_'.join([topo_src_name, str(stack_tile_id)])
    ext = "covars_cog.tif" 
    dem_cog_fn = os.path.join(tmp_out_path, "_".join([tileid, ext]))
    write_cog(mosaic, dem_cog_fn, sources[0].crs, out_trans, ["elevation"], out_crs=tile_parts['tile_crs'], resolution=(res, res))
    mosaic=None
    
    #
    # Call function to make all the topo covars for the output stk and write topo covars COG
    #     
    topo_stack_cog_fn = os.path.join(os.path.splitext(dem_cog_fn)[0] + '_topo_stack.tif')
    if args.output_dir is not None:
        topo_stack_cog_fn = os.path.join(output_dir, os.path.split(os.path.splitext(dem_cog_fn)[0])[1] + '_topo_stack.tif')
    topo_stack, topo_stack_names = make_topo_stack_cog(dem_cog_fn, topo_stack_cog_fn, tile_parts, res)
    print("Output topo covariate stack COG: ", topo_stack_cog_fn)

    return(topo_stack_cog_fn)

if __name__ == "__main__":
    main()