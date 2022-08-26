import os
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
import boto3
from typing import List
import argparse

import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.session import AWSSession 
from rasterio.warp import * #TODO: limit to specific needed modules
from rasterio.merge import merge
from rasterio.crs import CRS
from rio_tiler.models import ImageData
from rio_tiler.mosaic import mosaic_reader
from rio_tiler.io import COGReader

from CovariateUtils import write_cog, get_index_tile
from CovariateUtils_topo import *


def reader(src_path: str, bbox: List[float], epsg: CRS, dst_crs: CRS, height: int, width: int) -> ImageData:
    with COGReader(src_path) as cog:
        return cog.part(bbox, bounds_crs=epsg, max_size=None, dst_crs=dst_crs, height=height, width=width)

def get_shape(bbox, res=30):
    left, bottom, right, top = bbox
    width = int((right-left)/res)
    height = int((top-bottom)/res)
    return height,width

def build_stack_(stack_tile_fn: str, in_tile_id_col: str, stack_tile_id: str, tile_buffer_m: int, stack_tile_layer: str, covar_tile_fn: str, in_covar_s3_col: str, res: int, input_nodata_value: int, tmp_out_path: str, covar_src_name: str, clip: bool, topo_off: bool, output_dir: str):
    
    # Return the 4326 representation of the input <tile_id> geometry that is buffered in meters with <tile_buffer_m>
    tile_parts = get_index_tile(vector_path=stack_tile_fn, id_col=in_tile_id_col, tile_id=stack_tile_id, buffer=tile_buffer_m, layer=stack_tile_layer)
    #tile_parts = get_index_tile(stack_tile_fn, stack_tile_id, buffer=tile_buffer_m)
    geom_4326_buffered = tile_parts["geom_4326_buffered"]

    # Read the covar_tiles index file
    covar_tiles = gpd.read_file(covar_tile_fn)

    # intersect with the bbox tile
    covar_tiles_selection = covar_tiles.loc[covar_tiles.intersects(geom_4326_buffered.iloc[0])]

    # Get the s3 urls to the granules
    file_s3 = covar_tiles_selection[in_covar_s3_col].to_list()
    file_s3.sort()
    print("The covariate's filename(s) intersecting the {} m bbox for tile id {}:\n".format(str(tile_buffer_m), str(stack_tile_id)), '\n'.join(file_s3))

    # Create a mosaic from all the images
    bbox = tile_parts['geom_orig_buffered'].bounds.iloc[0].to_list()
    print(f"bbox: {bbox}")
    height, width = get_shape(bbox, res)

    img = mosaic_reader(file_s3, reader, bbox, tile_parts['tile_crs'], tile_parts['tile_crs'], height, width) 
    mosaic = (img[0].as_masked())
    out_trans =  rasterio.transform.from_bounds(*bbox, width, height)

    #
    # Writing tmp elevation COG so that we can read it in the way we need to (as a gdal.Dataset)
    #
    if (not os.path.isdir(tmp_out_path)): os.mkdir(tmp_out_path)
    tileid = '_'.join([covar_src_name, str(stack_tile_id)])
    ext = "cog.tif" 
    if not topo_off:
        cog_fn = os.path.join(tmp_out_path, "_".join([tileid, ext]))
    else:
        cog_fn = os.path.join(output_dir, "_".join([tileid, ext]))
    print(f'Writing stack as cloud-optimized geotiff: {cog_fn}')

    if clip:
        print("Clipping to feature polygon...")
        clip_geom = tile_parts['geom_orig']
        clip_crs = tile_parts['tile_crs']
    else:
        clip_geom = None
        clip_crs = None

    if not topo_off:
        bandnames_list = ["elevation"]
    else:
        bandnames_list = [covar_src_name]

    write_cog(mosaic, 
              cog_fn, 
              tile_parts['tile_crs'], 
              out_trans, 
              bandnames_list, 
              out_crs=tile_parts['tile_crs'], 
              resolution=(res, res), 
              clip_geom=clip_geom, 
              clip_crs=clip_crs, 
              input_nodata_value=input_nodata_value
             )

    mosaic=None

    if not topo_off:
        #
        # Call function to make all the topo covars for the output stk and write topo covars COG
        #     
        topo_stack_cog_fn = os.path.join(os.path.splitext(cog_fn)[0] + '_topo_stack.tif')
        if output_dir is not None:
            topo_stack_cog_fn = os.path.join(output_dir, os.path.split(os.path.splitext(cog_fn)[0])[1] + '_topo_stack.tif')
        topo_stack, topo_stack_names = make_topo_stack_cog(cog_fn, topo_stack_cog_fn, tile_parts, res)
        print(f"Output topo covariate stack COG: {topo_stack_cog_fn}")
        print(f"Removing tmp file: {cog_fn}")
        os.remove(cog_fn)
        os.remove(cog_fn + '.msk')

        return(topo_stack_cog_fn)
    else:
        return(cog_fn)


def main():
    '''Command line script to create topo stacks by vector tile id.
    python 3.1.5_dps.py --in_tile_fn /projects/shared-buckets/nathanmthomas/boreal_tiles_v003.gpkg --in_tile_num 1793 --tile_buffer_m 150 --in_tile_layer "boreal_tiles_v003" -o /projects/my-private-bucket/3.1.5_test/ --topo_tile_fn /projects/shared-buckets/nathanmthomas/dem30m_tiles.geojson

    example cmd line call: python 3.1.5_dps.py --in_tile_fn '/projects/shared-buckets/nathanmthomas/boreal_tiles_v003.gpkg' --in_tile_num 18822 --tile_buffer_m 120 --in_tile_layer "boreal_tiles_v003" -o '/projects/tmp/Topo/'

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_tile_fn", type=str, help="The input filename of a set of vector tiles that will define the bounds for stack creation")
    parser.add_argument("-n", "--in_tile_num", type=int, help="The id number of an input vector tile that will define the bounds for stack creation")
    parser.add_argument("-b", "--tile_buffer_m", type=int, default=None, help="The buffer size (m) applied to the extent of the specified stack tile")
    parser.add_argument("--in_tile_id_col", type=str, default="tile_num", help="The column of the tile layer name of the stack tiles dataset that holds the tile num")
    parser.add_argument("-l", "--in_tile_layer", type=str, default=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The path for the output stack")
    parser.add_argument("-r", "--res", type=int, default=30, help="The output resolution of the stack")
    parser.add_argument("-covar", "--covar_tile_fn", type=str, default="/projects/shared-buckets/nathanmthomas/dem30m_tiles.geojson", help="The filename of the covariates's set of vector tiles")
    parser.add_argument("--input_nodata_value", type=int, default=None, help="The input's nodata value")
    parser.add_argument("--in_covar_s3_col", type=str, default="s3", help="The column name that holds the s3 path of each s3 COG")
    parser.add_argument("-tmp", "--tmp_out_path", type=str, default="/projects/tmp", help="The tmp out path for the clipped covar cog before covar calcs")
    parser.add_argument("-name", "--covar_src_name", type=str, default="Copernicus", help="Name to identify the general source of the covariate data")
    parser.add_argument('--topo_off', dest='topo_off', action='store_true', help='Topo stack creation is a special case. Turn off topo stack covar extraction.')
    parser.set_defaults(topo_off=False)
    parser.add_argument('--clip', dest='clip', action='store_true', help='Clip to geom of feature id (tile or polygon)')
    parser.set_defaults(clip=False)

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
        
    # Parse all the args
    stack_tile_fn = args.in_tile_fn
    stack_tile_id = args.in_tile_num
    stack_tile_layer = args.in_tile_layer
    output_dir = args.output_dir
    in_tile_id_col = args.in_tile_id_col
    res = args.res
    tile_buffer_m = args.tile_buffer_m
    covar_tile_fn = args.covar_tile_fn
    input_nodata_value = args.input_nodata_value
    in_covar_s3_col = args.in_covar_s3_col
    tmp_out_path = args.tmp_out_path
    covar_src_name = args.covar_src_name
    topo_off = args.topo_off
    clip = args.clip
    
    print("\n---Running build_stack()---\n")
    build_stack_(stack_tile_fn, 
                in_tile_id_col, 
                stack_tile_id, 
                tile_buffer_m,
                stack_tile_layer, 
                covar_tile_fn,
                in_covar_s3_col,
                res,
                input_nodata_value,
                tmp_out_path,
                covar_src_name,
                clip, 
                topo_off,
                output_dir
               )

if __name__ == "__main__":
    main()