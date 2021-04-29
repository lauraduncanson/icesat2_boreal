#! /usr/bin/env python

import sys
import json
import os
from pprint import pprint
from osgeo import gdal
import boto3

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import numpy.ma as ma
from pyproj import Proj, Transformer

import geopandas as gpd
import shapely as shp
import folium
from shapely.geometry import box
from fiona.crs import from_epsg

import rasterio as rio
from rasterio.transform import Affine
from rasterio.session import AWSSession 
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import * #TODO: limit to specific needed modules
from rasterio.merge import merge
from rasterio import windows
from rasterio.crs import CRS

import argparse

from maap.maap import MAAP
maap = MAAP()

from CovariateUtils import write_cog, get_index_tile
from CovariateUtils_topo import *

def get_topo_stack(args):
    
    stack_tile_fn = args.stack_tile_fn
    stack_tile_id = args.stack_tile_id
    stack_tile_layer = args.stack_tile_layer
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
    topo_stack, topo_stack_names = make_topo_stack_cog(dem_cog_fn, topo_stack_cog_fn, tile_parts, res)
    print("Output topo covariate stack COG: ", topo_stack_cog_fn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-stk", "--stack_tile_fn", type=str, help="The filename of the stack's set of vector tiles")
    parser.add_argument("-stk_id", "--stack_tile_id", type=int, help="The specific id of a tile of the stack's tiles input that will define the bounds of the raster stacking")
    parser.add_argument("-buf", "--tile_buffer_m", type=int, default=None, help="The buffer size (m) applied to the extent of the specified stack tile")
    parser.add_argument("-lyr", "--stack_tile_layer", type=str, default=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-r", "--res", type=int, default=30, help="The output resolution of the stack")
    parser.add_argument("-topo", "--topo_tile_fn", type=str, default="/projects/maap-users/alexdevseed/dem30m_tiles.geojson", help="The filename of the topo's set of vector tiles")
    parser.add_argument("-tmp", "--tmp_out_path", type=str, default="/projects/tmp", help="The tmp out path for the clipped topo cog before topo calcs")
    parser.add_argument("-tsrc", "--topo_src_name", type=str, default="Copernicus", help="Name to identify the general source of the topography")

    args = parser.parse_args()
    
    if args.stack_tile_fn == None:
        print("Input a filename of the vector tiles that represents the arrangement by which the output stacks will be organized")
        os._exit(1)
    elif args.stack_tile_id == None:
        print("Input a specific tile id from the vector tiles the organize the stacks")
        os._exit(1)
    elif args.tile_buffer_m == None:
        print("Input a tile buffer distance in meters")
        os._exit(1)
    elif args.stack_tile_layer == None:
        print("Input a layer name from the stack tile vector file")
        os._exit(1)

    get_topo_stack(args)


if __name__ == "__main__":
    main()