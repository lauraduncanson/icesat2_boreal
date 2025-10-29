import os
from osgeo import gdal
import numpy as np
import numpy.ma as ma
import rasterio as rio
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio.warp import * #TODO: limit to specific needed modules
from rasterio.crs import CRS
from rio_cogeo.profiles import cog_profiles
from rio_tiler.utils import create_cutline
from rio_cogeo.cogeo import cog_translate

from CovariateUtils import write_cog, get_index_tile, get_shape

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

def make_topo_stack(dem_ma, tile_parts, res, do_scale=True):
    return None
    

def make_topo_stack_cog(dem_fn, topo_stack_cog_fn, tile_parts, res, do_scale=True, nodata_value=-9999):
    '''Calcs the topo covars, returns them as masked arrays, masks all where slope=0, stacks, clips, writes
    
    Note:
    do_scale: True (default) expect 'north-up' DEM data to slope on XY in degrees with Z in meters is scales correctly and it latitudinally specific
    '''
    print("\n\nMaking a topo stack...")
    print("Opening DEM...")
    dem_ds = gdal.Open(str(dem_fn))
    dem_ma = ds_getma(dem_ds)
    
    # Get input transform to correctly write_cog()
    # TODO : this src transform is no longer in the output tile projection - which is needed to write the output COG
    src_transform = Affine.from_gdal(*dem_ds.GetGeoTransform())
    pixelSizeX, pixelSizeY = (src_transform[0], -src_transform[4])
    print(f"Source res [x,y]:\t{pixelSizeX}, {pixelSizeY}")
    
    # Get output tiles transform
    #in_bbox = tile_parts['geom_orig_buffered'].bounds.iloc[0].to_list()
    in_bbox = tile_parts['bbox_orig']
    height, width = get_shape(in_bbox, res)
    out_trans =  rasterio.transform.from_bounds(*in_bbox, width, height)
    
    # Slope - need to ensure scale factor is correct - latitude-dependent
    # https://gis.stackexchange.com/questions/222378/working-with-dems-which-should-i-do-first-calculate-slope-or-reproject
    if do_scale:
        print('Scaling slope calc. based on latitude...')
        # Get latitude from transform
        latitude = list(src_transform)[5]
        slope_ds = gdal.DEMProcessing('', dem_ds, 'slope', format='MEM', scale=111320. * np.cos(latitude*np.pi/180))
    else:
        print('Not scaling slope calc...')
        slope_ds = gdal.DEMProcessing('', dem_ds, 'slope', format='MEM')
        
    slope_ma = ds_getma(slope_ds)
    slope_ma = np.ma.masked_where(slope_ma==0, slope_ma)
    
    # TPI
    # TODO: update this to standardized multi-scale TPI
    print("Calculating TPI...")
    tpi_ds = gdal.DEMProcessing('', dem_ds, 'tpi', format='MEM')
    tpi_ma = ds_getma(tpi_ds)
    tpi_ma = np.ma.masked_where(slope_ma==0, tpi_ma)
    
    # TSRI - need to ensure that calc happens on original (north up) data and not reprojected (not north up)
    # topo solar radiation index (a transformation of aspect; 0 - 1) TSRI = 0.5−cos((π/180)(aspect−30))/2
    # Matasci et al 2018: https://doi.org/10.1016/j.rse.2017.12.020
    print("Calculating TSRI...")
    aspect_ds = gdal.DEMProcessing('', dem_ds, 'aspect', format='MEM')
    aspect_ma = ds_getma(aspect_ds)
    tsri_ma = 0.5 - np.cos((np.pi/180.) * (aspect_ma - 30.))/2.
    tsri_ma = np.ma.masked_where(tsri_ma==nodata_value, tsri_ma)
    
    # Try this for removing water
    valid_mask = water_slope_mask(slope_ma)
    
    # Create topo_covar file list <-- below is actually a masked array list, not a file list
    topo_covar_file_list = [dem_ma, slope_ma, tsri_ma, tpi_ma, valid_mask]
    
    # Close the gdal datasets
    dem_ds = slope_ds = aspect_ds = tpi_ds = None
    
    # Stack
    # move axis of the stack so bands is first
    topo_stack = np.transpose([dem_ma, slope_ma, tsri_ma, tpi_ma, valid_mask], [0,1,2])
    topo_stack_names = ["elevation","slope","tsri","tpi", "slopemask"]
    
    print("Write COG (w/ reprj), read COG (with clip), write final COG...")
    ######    
    # 1st write using write_cog()
    # Note: this assumes array is in 4326 (good assumption since this is topo data)
    out_tmp_fn = os.path.splitext(topo_stack_cog_fn)[0]+'_tmp.tif'
    print("Writing a tmp stack w/ reproj and a clip with 4326 geom...")
    write_cog(topo_stack, 
              out_tmp_fn, 
              tile_parts['geom_4326'].crs, # Here you need the covar (in) tiles crs
              src_transform,
              topo_stack_names, 
              out_crs = tile_parts['geom_orig'].crs, # CURRENT WORKING
              ####out_crs = tile_parts['geom_4326'].crs,     # CURRENT TESTING
              clip_geom = tile_parts['geom_4326'],    
              clip_crs = tile_parts['geom_4326'].crs, #
              resolution = (res, res),
              align = True, 
              input_nodata_value=nodata_value,
              resampling='cubic'
             )

    ######    
    # Read COG back in and clip (geom_orig - the geom of the projected tile system into which you put data)
    from rio_tiler.io import COGReader
    from rasterio.crs import CRS
    from typing import List
    from rio_tiler.models import ImageData

    def reader(src_path: str, bbox: List[float], epsg: CRS, dst_crs: CRS, height: int, width: int) -> ImageData:
        with COGReader(src_path) as cog:
            print('Image read and clipped:\t\t', src_path)
            return cog.part(bbox, bounds_crs=epsg, max_size=None, dst_crs=dst_crs, height=height, width=width)

    bbox = tile_parts['geom_orig'].bounds.iloc[0].to_list()
    height, width = get_shape(bbox, res)

    img = reader(out_tmp_fn, bbox, tile_parts['geom_orig'].crs, tile_parts['geom_orig'].crs, height, width) # CURRENT WORKING
    ####img = reader(out_tmp_fn, bbox, tile_parts['geom_4326'].crs, tile_parts['geom_orig'].crs, height, width)     # CURRENT TESTING
    #topo_stack = img.as_masked().data
    topo_stack = img.array

    ######
    # 2nd write (to a COG), this time the data is already clipped and reprojected - so neither of those should be needed - but they are when align=true.
    print(f"Current stack shape:\t\t({img.count},{img.width},{img.height})")
    # Translate to a COG
    # Get the rio-cogeo profile for deflate compression, modify some of the options
    dst_profile = cog_profiles.get("deflate")
    dst_profile['blockxsize']=256
    dst_profile['blockysize']=256
    dst_profile['predictor']=1 # originally set to 2, which fails with 'int'; 1 tested successfully for 'int' and 'float64'
    dst_profile['zlevel']=7
    dst_profile['height']=img.height
    dst_profile['width']=img.width
    dst_profile['count']=img.count
    dst_profile['dtype']=str(topo_stack.dtype)
    dst_profile['crs']=img.crs
    dst_profile['resolution']=res
    dst_profile['transform']=out_trans
    dst_profile['nodata']=nodata_value
    with rio.open(topo_stack_cog_fn, 'w+', **dst_profile) as final_cog:
        final_cog.descriptions = tuple(topo_stack_names)
        final_cog.write(topo_stack)
        #cog_translate(final_cog, topo_stack_cog_fn, dst_profile, overview_resampling='cubic')
    cog_translate(topo_stack_cog_fn, topo_stack_cog_fn, dst_profile, overview_resampling='cubic')
    print('Image written to disk:\t\t', topo_stack_cog_fn)

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