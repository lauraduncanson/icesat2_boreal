import os
import rasterio
#os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
#os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
import boto3
import s3fs
import requests
from typing import List
import argparse

import numpy as np

import rasterio
import rasterio as rio
from rasterio.session import AWSSession 
from rasterio.warp import * #TODO: limit to specific needed modules
from rasterio.merge import merge
from rasterio.crs import CRS
from rio_tiler.models import ImageData
from rio_tiler.mosaic import mosaic_reader
from rio_tiler.mosaic.methods import defaults
from rio_tiler.io import COGReader, Reader

import CovariateUtils
from CovariateUtils import write_cog, get_index_tile, get_shape, reader
from CovariateUtils_topo import *

from maap.maap import MAAP
maap = MAAP(maap_host='api.maap-project.org')

#Check for file existence
def fn_check(fn):
    """Wrapper to check for file existence
    
    Parameters
    ----------
    fn : str
        Input filename string.
    
    Returns
    -------
    bool
        True if file exists, False otherwise.
    """
    return os.path.exists(fn)

def fn_check_full(fn):
    """Check for file existence
    Avoids race condition, but slower than os.path.exists.
    
    Parameters
    ----------
    fn : str
        Input filename string.
    
    Returns
    -------
    status 
        True if file exists, False otherwise.
    """
    status = True 
    if not os.path.isfile(fn): 
        status = False
    else:
        try: 
            open(fn) 
        except IOError:
            status = False
    return status

def fn_list_check(fn_list):
    status = True
    for fn in fn_list:
        if not fn_check(fn):
            print('Unable to find: %s' % fn)
            status = False
    return status

def fn_list_valid(fn_list):
    print('%i input fn' % len(fn_list))
    out_list = []
    for fn in fn_list:
        if not fn_check(fn):
            print('Unable to find: %s' % fn)
        else:
            out_list.append(fn)
    print('%i output fn' % len(out_list))
    return out_list 

s3_cred_endpoint = {
    'podaac':   'https://archive.podaac.earthdata.nasa.gov/s3credentials',
    'gesdisc':  'https://data.gesdisc.earthdata.nasa.gov/s3credentials',
    'lpdaac':   'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials',
    'ornldaac': 'https://data.ornldaac.earthdata.nasa.gov/s3credentials',
    'ghrcdaac': 'https://data.ghrc.earthdata.nasa.gov/s3credentials'
}

def get_temp_creds(provider):
    return requests.get(s3_cred_endpoint[provider]).json()

def build_stack_(stack_tile_fn: str, in_tile_id_col: str, stack_tile_id: str, tile_buffer_m: int, stack_tile_layer: str, covar_tile_fn: str, in_covar_s3_col: str, res: int, input_nodata_value: int, tmp_out_path: str, covar_src_name: str, bandnames_list: list, clip: bool, topo_off: bool, output_dir: str, height: None, width: None, band_indexes_list: list):
    
    # Return the 4326 representation of the input <tile_id> geometry that is buffered in meters with <tile_buffer_m>
    tile_parts = get_index_tile(vector_path=stack_tile_fn, id_col=in_tile_id_col, tile_id=stack_tile_id, buffer=tile_buffer_m, layer=stack_tile_layer)
    #tile_parts = get_index_tile(stack_tile_fn, stack_tile_id, buffer=tile_buffer_m)
    geom_4326_buffered = tile_parts["geom_4326_buffered"]

    # Read the covar_tiles index file
    covar_tiles = gpd.read_file(covar_tile_fn).to_crs(4326)

    # intersect with the bbox tile
    covar_tiles_selection = covar_tiles.loc[covar_tiles.intersects(geom_4326_buffered.iloc[0])]

    # Get the s3 urls to the granules
    files_list_s3 = covar_tiles_selection[in_covar_s3_col].to_list()
    files_list_s3.sort()
    print(f"{len(files_list_s3)} covariate filename(s) intersecting the {tile_buffer_m} m buffered bbox for tile id {stack_tile_id}:\n")
    
    # Create a mosaic from all the images
    in_bbox = tile_parts['geom_orig_buffered'].bounds.iloc[0].to_list()
    print(f"in_bbox: {in_bbox}")
    
    # This is added to allow the output size to be forced to a certain size - this avoids have some tiles returned as 2999 x 3000 due to rounding issues.
    # Most tiles dont have this problem and thus dont need this forced shape, but some consistently do. 
    if height is None or not topo_off:
        # Resets height, width based on bbox computerd with tile buffer
        # Needed to handle topo runs
        print(f'Getting output height and width from buffered (buffer={tile_buffer_m}) original tile geometry...')
        height, width = get_shape(in_bbox, res)
    else:
        print('Getting output height and width from input shape arg...')
        
    print(f"{height} x {width}")

    #
    # Mosiac: use rio_tiler to read in a list of files and return a mosaic
    #
    
    # Get the band indexes associated with the band names of interest
    if band_indexes_list is None:
        band_indexes_list = list(range(1, len(bandnames_list) + 1)) #[1,2,3,4,7,8]# get_band_indices(files_list_s3[0], bandnames_list)
    print(f'Band indexes list: {band_indexes_list}')
    bandnames_list = [bandnames_list[i-1] for i in band_indexes_list]
    print(bandnames_list)
    
    print('\n'.join(files_list_s3))
    
    #########################
    # Get rio Env aws session
    
    # Setup a dummy aws env session for rio
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
    rio_env_session = rio.Env(rasterio.session.AWSSession())
    
    # Set up AWS session according to strings in file list
    if 'ornl' in files_list_s3[0]:
        os.environ['AWS_NO_SIGN_REQUEST'] = 'NO'
        print('Accessing ORNL DAAC data...')
        rio_env_session = rio.Env(CovariateUtils.get_aws_session_DAAC(maap.aws.earthdata_s3_credentials(CovariateUtils.s3_cred_endpoint_DAAC['ornldaac'])))
        
    with rio_env_session:
        # updated call to manage memory and accomodate subsetting by bands (indexes)
        # works for ~56 files of float COGs with arrays of 9 x ~700 x ~700 to return mosaic COGs with arrays of 9 x 3000 x 3000
        MAX_FILES = 30
        NUM_THREADS_MOSAIC = 5
        
        if len(files_list_s3) > MAX_FILES:
            print(f' ~~Entering memory management mode to complete run on 32 GB worker~~\nReducing threads to {NUM_THREADS_MOSAIC} since # files > {MAX_FILES}...')
            img = mosaic_reader(files_list_s3, reader, in_bbox, tile_parts['tile_crs'], tile_parts['tile_crs'], height, width, band_indexes_list, threads=NUM_THREADS_MOSAIC, pixel_selection=defaults.HighestMethod())
        else:
            img = mosaic_reader(files_list_s3, reader, in_bbox, tile_parts['tile_crs'], tile_parts['tile_crs'], height, width, band_indexes_list)
        
    mosaic = (img[0].as_masked())
    
    print(f'Stack (mosaic) shape: {mosaic.shape}')
    out_trans =  rasterio.transform.from_bounds(*in_bbox, width, height)

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

    # Don't clip yet if doing a topo run 
    DO_ALIGN=True
    if topo_off and clip:
        print("Clipping to feature polygon...")
        clip_geom = tile_parts['geom_orig']
        clip_crs = tile_parts['tile_crs']
    else:
        # Do not clip and align what
        # for topo runs that need extra pixels outside of tile_parts['geom_orig'] in order to calc slope, tsri, etc
        # tile_parts needed for final clip is still passed to write_cog() via make_topo_stack()
        clip_geom = None
        clip_crs = None
        DO_ALIGN = False

    if not topo_off:
        bandnames_list = ["elevation"]
    #else:
    #   bandnames_list = covar_src_name_list

    write_cog(mosaic, 
              cog_fn, 
              tile_parts['tile_crs'], 
              out_trans, 
              bandnames_list, 
              out_crs=tile_parts['tile_crs'], 
              resolution=(res, res), 
              clip_geom=clip_geom, 
              clip_crs=clip_crs, 
              align=DO_ALIGN, # manually turn this on for some testing
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

def build_stack_list(covar_dict_list, vector_dict,
                     #stack_tile_fn: str, in_tile_id_col: str, stack_tile_id: str, 
                     tile_buffer_m: int, 
                     #stack_tile_layer: str, 
                     #covar_tile_fn, in_covar_s3_col: str, 
                     res: int, 
                     #input_nodata_value: int, tmp_out_path: str, covar_src_name: str, 
                     clip: bool, 
                     #topo_off: bool, 
                     output_dir: str, 
                     height: None, width: None,
                     MAKE_DF=False
                    ):
    '''
    Build a stack of a list of data and export as a multi-band COG
    TODO:  in_covar_s3_col is s3_path for each covar_tile_fn; to vary this by fn, make this var into a list that corresponds with covar_tile_fn_list
    '''
    stack_tile_fn =   vector_dict['INDEX_FN']
    in_tile_id_col =  vector_dict['ID_COL_NAME']
    stack_tile_id =   vector_dict['TILE_NUM']
    stack_tile_layer = vector_dict['INDEX_LYR']
    
    ext = "cog.tif" 
    ## Check file list
    #covar_tile_fn_list = fn_list_valid(covar_tile_fn_list)
    
    if len(covar_dict_list) == 0:
        print('\nNo valid files in build_stack_list. Exiting.\n')
        os.exit(1)
    else:
            
        # Return the 4326 representation of the input <tile_id> geometry that is buffered in meters with <tile_buffer_m>
        tile_parts = get_index_tile(vector_path=stack_tile_fn, id_col=in_tile_id_col, tile_id=stack_tile_id, buffer=tile_buffer_m, layer=stack_tile_layer)
        #tile_parts = get_index_tile(stack_tile_fn, stack_tile_id, buffer=tile_buffer_m)
        geom_4326_buffered = tile_parts["geom_4326_buffered"]
        
        print('Looping through file build_stack_list...')
        stack_tile_mosaic_list = []
        stack_bandnames_list = []
        
        for covar_dict in covar_dict_list:
            
            # these used to be indiv args; now come from input dict
            covar_tile_fn = covar_dict['COVAR_TILE_FN']
            covar_src_name = covar_dict['RASTER_NAME'] # incomlpete - should update to make this a list to accomodate multiband covar files [0] # Changed to 1st element of input list
            in_covar_s3_col = covar_dict['IN_COVAR_S3_COL']
            input_nodata_value = covar_dict['NODATA_VAL']

            # Read the covar_tiles index file
            covar_tiles = gpd.read_file(covar_tile_fn)

            # intersect with the bbox tile
            covar_tiles_selection = covar_tiles.loc[covar_tiles.intersects(geom_4326_buffered.iloc[0])]

            # Get the s3 urls to the granules
            files_list_s3 = covar_tiles_selection[in_covar_s3_col].to_list()
            files_list_s3.sort()
            print("The covariate's filename(s) intersecting the {} m buffered bbox for tile id {}:\n".format(str(tile_buffer_m), str(stack_tile_id)), '\n'.join(files_list_s3))

            # Create a mosaic from all the images
            in_bbox = tile_parts['geom_orig_buffered'].bounds.iloc[0].to_list()
            print(f"in_bbox: {in_bbox}")

            # This is added to allow the output size to be forced to a certain size - this avoids have some tiles returned as 2999 x 3000 due to rounding issues.
            # Most tiles dont have this problem and thus dont need this forced shape, but some consistently do. 
            if height is None:
                print(f'Getting output height and width from buffered (buffer={tile_buffer_m}) original tile geometry...')
                height, width = get_shape(in_bbox, res)
            else:
                print('Getting output height and width from input shape arg...')

            print(f"{height} x {width}")

            img = mosaic_reader(files_list_s3, reader, in_bbox, tile_parts['tile_crs'], tile_parts['tile_crs'], height, width) 
            mosaic = (img[0].as_masked())
            out_trans =  rasterio.transform.from_bounds(*in_bbox, width, height)
            
            if clip:
                print("Clipping to feature polygon...")
                clip_geom = tile_parts['geom_orig']
                clip_crs = tile_parts['tile_crs']
            else:
                clip_geom = None
                clip_crs = None

            bandnames_list = [covar_src_name]
            print(f'mosaic shape: {mosaic.shape}')
            stack_bandnames_list.append(covar_src_name)
            stack_tile_mosaic_list.append(mosaic)
            
    # Stack bands together
    print("\nCreating raster stack...\n")
    # These must correspond with the bandnames
    stack = np.vstack(stack_tile_mosaic_list)  
    print(f'stack shape: {stack.shape}')
    # New output COG name 
    cog_fn = os.path.join(output_dir, "_".join(['build_stack', str(stack_tile_id), ext]))
    print(f'Writing stack as cloud-optimized geotiff: {cog_fn}')
    write_cog(stack, 
              cog_fn, 
              tile_parts['tile_crs'], 
              out_trans, 
              stack_bandnames_list, 
              out_crs=tile_parts['tile_crs'], 
              resolution=(res, res), 
              clip_geom=clip_geom, 
              clip_crs=clip_crs, 
              align=True, # manually turn this on for some testing
              input_nodata_value=input_nodata_value
             )

    mosaic=None
    if MAKE_DF:
        csv_fn = cog_fn.split('_'+ext)[0] + '.csv'
        print(f'Writing a dataframe csv: {csv_fn}')
        # https://gis.stackexchange.com/questions/361318/create-pandas-dataframe-from-raster-image-one-row-per-pixel-with-bands-as-colu
        pd.DataFrame(stack.reshape([len(covar_dict_list),-1]).T, columns=[d['RASTER_NAME'] for d in covar_dict_list]).to_csv(csv_fn)
    return(cog_fn)
    

def main():
    '''Build COG stacks (multiple bands) for an extent based on a vector polygon.
       uses rio_tiler to read and mosaic input lists of COGs stored on s3
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
    parser.add_argument("-covar", "--covar_tile_fn", type=str, default="/projects/shared-buckets/nathanmthomas/dem30m_tiles.geojson", help="The filename of the covariates's set of vector tiles")
    parser.add_argument("--input_nodata_value", type=int, default=None, help="The input's nodata value")
    parser.add_argument("--in_covar_s3_col", type=str, default="s3", help="The column name that holds the s3 path of each s3 COG")
    parser.add_argument("-tmp", "--tmp_out_path", type=str, default="/tmp", help="The tmp out path for the clipped covar cog before covar calcs")
    parser.add_argument("-name", "--covar_src_name", type=str, default="Copernicus", help="Name to identify the general source of the covariate data")
# https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser.add_argument("--bandnames_list", type=str, nargs='+', action='store', default=['elevation'], help="List of names to identify the bandnames")
    parser.add_argument("--band_indexes_list", type=int, nargs='+', action='store', default=None, help="List of indexes (starting from 1) to identify the bands to be subsetted. If None, then all input bands are returned.")
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
    height = args.shape
    width = args.shape
    tile_buffer_m = args.tile_buffer_m
    covar_tile_fn = args.covar_tile_fn
    input_nodata_value = args.input_nodata_value
    in_covar_s3_col = args.in_covar_s3_col
    tmp_out_path = args.tmp_out_path
    bandnames_list = args.bandnames_list
    band_indexes_list = args.band_indexes_list
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
                bandnames_list,
                clip, 
                topo_off,
                output_dir,
                height,
                width,
                band_indexes_list
               )

if __name__ == "__main__":
    main()