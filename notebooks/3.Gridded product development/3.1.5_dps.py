
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

from maap.maap import MAAP
maap = MAAP()

from CovariateUtils import write_cog, get_index_tile

# Functions 
# copied from pygeotools/iolib.py TODO: full citation
# Q: direct import, but pygeotools isn't in conda-forge OR reimplement

#Given input dataset, return a masked array for the input band
def ds_getma(ds, bnum=1):
    """Get masked array from input GDAL Dataset
    Parameters
    ----------
    ds : gdal.Dataset 
        Input GDAL Datset
    bnum : int, optional
        Band number
    
    Returns
    -------
    np.ma.array    
        Masked array containing raster values
    """
    b = ds.GetRasterBand(bnum)
    return b_getma(b)

#Given input band, return a masked array
def b_getma(b):
    """Get masked array from input GDAL Band
    Parameters
    ----------
    b : gdal.Band 
        Input GDAL Band 
    
    Returns
    -------
    np.ma.array    
        Masked array containing raster values
    """
    b_ndv = get_ndv_b(b)
    #bma = np.ma.masked_equal(b.ReadAsArray(), b_ndv)
    #This is more appropriate for float, handles precision issues
    bma = np.ma.masked_values(b.ReadAsArray(), b_ndv)
    return bma

#Return nodata value for GDAL band
def get_ndv_b(b):
    """Get NoData value for GDAL band.
    If NoDataValue is not set in the band, 
    extract upper left and lower right pixel values.
    Otherwise assume NoDataValue is 0.
 
    Parameters
    ----------
    b : GDALRasterBand object 
        This is the input band.
 
    Returns
    -------
    b_ndv : float 
        NoData value 
    """

    b_ndv = b.GetNoDataValue()
    if b_ndv is None:
        #Check ul pixel for ndv
        ns = b.XSize
        nl = b.YSize
        ul = float(b.ReadAsArray(0, 0, 1, 1))
        #ur = float(b.ReadAsArray(ns-1, 0, 1, 1))
        lr = float(b.ReadAsArray(ns-1, nl-1, 1, 1))
        #ll = float(b.ReadAsArray(0, nl-1, 1, 1))
        #Probably better to use 3/4 corner criterion
        #if ul == ur == lr == ll:
        if np.isnan(ul) or ul == lr:
            b_ndv = ul
        else:
            #Assume ndv is 0
            b_ndv = 0
    elif np.isnan(b_ndv):
        b_dt = gdal.GetDataTypeName(b.DataType)
        if 'Float' in b_dt:
            b_ndv = np.nan
        else:
            b_ndv = 0
    print("NoData Value: ",b_ndv)
    return b_ndv

def water_slope_mask(slope_ma):
    m = np.zeros_like(slope_ma)
    m = np.where(slope_ma > 0, 1, m)
    print("Slope mask created to indicate water where slope = 0")
    return m

def make_topo_stack_cog(dem_fn, topo_stack_cog_fn, tile_parts, out_transform):
    '''Calcs the topo covars, returns them as masked arrays, masks all where slope=0, stacks, clips, writes
    '''
    print("Opening DEM...")
    dem_ds = gdal.Open(str(dem_fn))
    dem_ma = ds_getma(dem_ds)
    src_transform = Affine.from_gdal(*dem_ds.GetGeoTransform())
    print(src_transform)

    # Slope
    print("Calculating Slope...")
    slope_ds = gdal.DEMProcessing('', dem_ds, 'slope', format='MEM')
    slope_ma = ds_getma(slope_ds)
    slope_ma = np.ma.masked_where(slope_ma==0, slope_ma)
    
    # TPI
    # TODO: update this to standardized multi-scale TPI
    print("Calculating TPI...")
    tpi_ds = gdal.DEMProcessing('', dem_ds, 'tpi', format='MEM')
    tpi_ma = ds_getma(tpi_ds)
    tpi_ma = np.ma.masked_where(slope_ma==0, tpi_ma)
    
    # TSRI
    # topo solar radiation index (a transformation of aspect; 0 - 1) TSRI = 0.5−cos((π/180)(aspect−30))/2
    # Matasci et al 2018: https://doi.org/10.1016/j.rse.2017.12.020
    print("Calculating TSRI...")
    aspect_ds = gdal.DEMProcessing('', dem_ds, 'aspect', format='MEM')
    aspect_ma = ds_getma(aspect_ds)
    tsri_ma = 0.5 - np.cos((np.pi/180.) * (aspect_ma - 30.))/2.
    tsri_ma = np.ma.masked_where(tsri_ma==-9999, tsri_ma)
    
    # Try this for removing water
    valid_mask = slopeMask(slope_ma)
    
    # Create topo_covar file list <-- below is actually a masked array list, not a file list
    topo_covar_file_list = [dem_ma, slope_ma, tsri_ma, tpi_ma, valid_mask]
    
    # Close the gdal datasets
    dem_ds = slope_ds = aspect_ds = tpi_ds = None
    
    # TODO: CLip stack to in_bbox
    # TODO: Mask the file with the in_bbox?! - MaskArrays wants a file list, not a masked array list (which is what I have given it...)
    #topo_covar_file_list_bbox = [MaskArrays(topo_covar_file_list[i], in_bbox) for i in range(len(topo_covar_file_list))]
    
    # Stack
    # move axis of the stack so bands is first
    topo_stack = np.transpose([dem_ma, slope_ma, tsri_ma, tpi_ma, valid_mask], [0,1,2])
    
    #topo_stack = np.transpose(topo_covar_file_list_bbox, [0,1,2])
    topo_stack_names = ["elevation","slope","tsri","tpi", "slopemask"]
    
    # TODO: to be safe out_crs should be read from the gdal dataset
    write_cog(topo_stack, 
              topo_stack_cog_fn, 
              tile_parts['tile_crs'], 
              out_transform, 
              topo_stack_names, 
              clip_geom = tile_parts['geom_orig'],
              clip_crs = tile_parts['tile_crs'], 
              align = True)
    
    return(topo_stack, topo_stack_names)

def get_topo_stack(stack_tile_fn, stack_tile_id, stack_tile_layer, tile_buffer_m, topo_tile_fn):
    
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
    # TODO: test passing all s3 sources
    mosaic, out_trans = merge(sources, bounds = geom_4326_buffered)
    
    #
    # Writing tmp elevation COG so that we can read 
    #
    if (not os.path.isdir(tmp_out_path)): os.mkdir(tmp_out_path)
    tileid = '_'.join([topo_src_name, str(bbox_ID)])
    ext = "covars_cog.tif" 
    dem_cog_fn = os.path.join(tmp_out_path, "_".join([tileid, ext]))
    write_cog(mosaic, dem_cog_fn, sources[0].crs, out_trans, ["elevation"], out_crs=tile_index['tile_crs'])
    
    #
    # Call function to make all the topo covars for the output stk and write topo covars COG
    #
    topo_stack_cog_fn = os.path.join(os.path.splitext(dem_cog_fn)[0] + '_topo_stack.tif')
    topo_stack, topo_stack_names = make_topo_stack_cog(dem_cog_fn, topo_stack_cog_fn)
    print("Output topo covariate")
    return()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-stk", "--stack_tile_fn", type=str, help="The filename of the stack's set of vector tiles")
    parser.add_argument("-stk_id", "--stack_tile_id", type=int, help="The specific id of a tile of the stack's tiles input that will define the bounds of the raster stacking")
    parser.add_argument("-buf", "--tile_buffer_m", type=int, default=None, help="The buffer size (m) applied to the extent of the specified stack tile")
    parser.add_argument("-lyr", "--stack_tile_layer", type=str, defaut=None, help="The layer name of the stack tiles dataset")
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

    # tile_buffer_m = 120
    # stack_tile_layer = "boreal_tiles_albers"
    get_topo_stack(args)


if __name__ == "__main__":
    main()