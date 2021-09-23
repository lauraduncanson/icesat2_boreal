#import sys
#import json
import os

from osgeo import gdal

#import boto3

#import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow
import numpy as np
import numpy.ma as ma
#from pyproj import Proj, Transformer



#import geopandas as gpd
#import shapely as shp
#import folium
#from shapely.geometry import box
#from fiona.crs import from_epsg
import rasterio as rio
from rasterio.transform import Affine
#from rasterio.session import AWSSession 
#from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import * #TODO: limit to specific needed modules
#from rasterio.merge import merge
#from rasterio import windows
#from rasterio.io import MemoryFile
from rasterio.crs import CRS
#from rasterio.vrt import WarpedVRT
from rio_cogeo.profiles import cog_profiles
from rio_tiler.utils import create_cutline
from rio_cogeo.cogeo import cog_translate

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

def hillshade(array,azimuth,angle_altitude):
    #https://github.com/rveciana/introduccion-python-geoespacial/blob/master/hillshade.py
    azimuth = 360.0 - azimuth 

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)

    return 255*(shaded + 1)/2

def water_slope_mask(slope_ma):
    m = np.zeros_like(slope_ma)
    m = np.where(slope_ma > 0, 1, m)
    print("Slope mask created to indicate water where slope = 0")
    return m

def make_topo_stack_cog(dem_fn, topo_stack_cog_fn, tile_parts, res):
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
    valid_mask = water_slope_mask(slope_ma)
    
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
              src_transform,
              topo_stack_names, 
              clip_geom = tile_parts['geom_orig'],
              clip_crs = tile_parts['tile_crs'],
              resolution = (res, res),
              align = True)
    
    return(topo_stack, topo_stack_names)

def hillshade(array,azimuth,angle_altitude):
    #https://github.com/rveciana/introduccion-python-geoespacial/blob/master/hillshade.py
    azimuth = 360.0 - azimuth 

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)

    return 255*(shaded + 1)/2