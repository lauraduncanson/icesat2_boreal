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

from CovariateUtils import write_cog, get_index_tile, get_shape, reader
from CovariateUtils_topo import *

# def reader(src_path: str, bbox: List[float], epsg: CRS, dst_crs: CRS, height: int, width: int) -> ImageData:
#     with COGReader(src_path) as cog:
#         return cog.part(bbox, bounds_crs=epsg, max_size=None, dst_crs=dst_crs, height=height, width=width)

# def get_shape(bbox, res=30):
#     left, bottom, right, top = bbox
#     width = int((right-left)/res)
#     height = int((top-bottom)/res)
#     return height,width
#     #return 3000,3000

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
    parser.add_argument("--shape", type=int, default=None, help="The output height and width of the grid's shape. If None, get from input tile.")    
    parser.add_argument("-topo", "--topo_tile_fn", type=str, default="/projects/shared-buckets/nathanmthomas/dem30m_tiles.geojson", help="The filename of the topo's set of vector tiles")
    parser.add_argument("-tmp", "--tmp_out_path", type=str, default="/projects/tmp", help="The tmp out path for the clipped topo cog before topo calcs")
    parser.add_argument("-tsrc", "--topo_src_name", type=str, default="Copernicus", help="Name to identify the general source of the topography")
    parser.add_argument('--topo_off', dest='topo_off', action='store_true', help='Turn off topo stack covar extraction')
    parser.set_defaults(topo_off=False)

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
    tile_parts = get_index_tile(vector_path=stack_tile_fn, id_col=args.in_tile_id_col, tile_id=stack_tile_id, buffer=tile_buffer_m, layer = stack_tile_layer)
    #tile_parts = get_index_tile(stack_tile_fn, stack_tile_id, buffer=tile_buffer_m)
    geom_4326_buffered = tile_parts["geom_4326_buffered"]
    
    # Read the topography index file
    dem_tiles = gpd.read_file(topo_tile_fn)

    # intersect with the bbox tile
    dem_tiles_selection = dem_tiles.loc[dem_tiles.intersects(geom_4326_buffered.iloc[0])]


    # Set up and aws permissions to public bucket
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    
    # Get the s3 urls to the granules
    file_s3 = dem_tiles_selection["s3"].to_list()
    file_s3.sort()
    print("The DEM filename(s) intersecting the {} m buffered bbox for tile id {}:\n".format(str(tile_buffer_m), str(stack_tile_id)), '\n'.join(file_s3))
    
    # Create a mosaic from all the images
    in_bbox = tile_parts['geom_orig_buffered'].bounds.iloc[0].to_list()
    print(f"in_bbox: {in_bbox}")
    
    # This is added to allow the output size to be forced to a certain size - this avoids have some tiles returned as 2999 x 3000 due to rounding issues.
    # Most tiles dont have this problem and thus dont need this forced shape, but some consistently do. 
    if args.shape is None:
        print(f'Getting output height and width from buffered (buffer={tile_buffer_m}) original tile geometry...')
        height, width = get_shape(in_bbox, res)
    else:
        print('Getting output height and width from input shape arg...')
        height = args.shape
        width = args.shape
    
    print(f"{height} x {width}")
    
    img = mosaic_reader(file_s3, reader, in_bbox, tile_parts['tile_crs'], tile_parts['tile_crs'], height, width) 
    mosaic = (img[0].as_masked())
    out_trans =  rasterio.transform.from_bounds(*in_bbox, width, height)
    
    #
    # Writing tmp elevation COG so that we can read it in the way we need to (as a gdal.Dataset)
    #
    if (not os.path.isdir(tmp_out_path)): os.mkdir(tmp_out_path)
    tileid = '_'.join([topo_src_name, str(stack_tile_id)])
    ext = "cog.tif" 

    cog_fn = os.path.join(tmp_out_path, "_".join([tileid, ext]))
    write_cog(mosaic, cog_fn, tile_parts['tile_crs'], out_trans, ["elevation"], out_crs=tile_parts['tile_crs'], resolution=(res, res))

    mosaic=None
    
    if not args.topo_off:
        #
        # Call function to make all the topo covars for the output stk and write topo covars COG
        #     
        topo_stack_cog_fn = os.path.join(os.path.splitext(cog_fn)[0] + '_topo_stack.tif')
        if args.output_dir is not None:
            topo_stack_cog_fn = os.path.join(output_dir, os.path.split(os.path.splitext(cog_fn)[0])[1] + '_topo_stack.tif')
        topo_stack, topo_stack_names = make_topo_stack_cog(cog_fn, topo_stack_cog_fn, tile_parts, res)
        print(f"Output topo covariate stack COG: {topo_stack_cog_fn}")
        print(f"Removing tmp file: {cog_fn}")
        os.remove(cog_fn)
        os.remove(cog_fn + '.msk')

        return(topo_stack_cog_fn)
    else:
        return(cog_fn)

if __name__ == "__main__":
    main()