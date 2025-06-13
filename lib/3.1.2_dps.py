import argparse
import os
import geopandas as gpd
import boto3
import rasterio as rio
import geopandas as gpd
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.warp import *
from rasterio.merge import merge
from rasterio.crs import CRS
from rio_tiler.io import COGReader
import numpy as np
from rasterio.session import AWSSession
from typing import List
from CovariateUtils import write_cog, get_index_tile, get_aws_session, get_aws_session_DAAC, common_mask, get_shape, reader
#from fetch_HLS import get_HLS_data, get_LC2SR_data, get_ms_data, MS_BANDS_DICT
from fetch_from_api import get_ms_data, MS_BANDS_DICT
import json
import datetime
from CovariateUtils import get_creds, get_creds_DAAC

from maap.maap import MAAP
maap = MAAP()

def get_json(s3path, output):
    '''
    Download a json from S3 to the output directory
    '''
    aws_session = boto3.session.Session()
    s3 = aws_session.resource('s3')
    output_file = os.path.join(output_dir, os.path.basename(s3path))
    #TODO split the bucket name from the s3 path
    bucket_name = s3path.split("/")[2]
    s3_key = "/".join(samples3.split("/")[3:])
    s3.Bucket(bucket_name).download_file(s3_key, output_file)
    
    with open(output_file) as f:
        catalog = json.load(f) 
    return catalog

def GetBandLists___(inJSON, bandnum, comp_type):
    
    BandList = []
    with open(inJSON) as f:
        response = json.load(f)
        
    for i in range(len(response['features'])):
        
        if comp_type=='HLS':
            # Get the HLS product type and the product-specific bands from each feature of the JSON
            product_type = response['features'][i]['id'].split('.')[1]
            if product_type=='L30':
                bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
            elif product_type=='S30':
                bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B8A', 6:'B11', 7:'B12',8:'Fmask'})
            else:
                print("HLS product type not recognized: Must be L30 or S30.")
                os._exit(1)
        elif comp_type in MS_BANDS_DICT.keys() and comp_type in ['LT05','LE07','LC08']:
            # Get the Landsat product type and the product-specific bands from each feature of the JSON
            product_type = response['features'][i]['id'].split('_')[0]
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22'})
        else:
            print(f"comp type ({comp_type}) not recognized")
            os._exit(1)
            
        try:
            #print(f'GetBandLists: {comp_type} {product_type} {bands[bandnum]}')
            getBand = response['features'][i]['assets'][bands[bandnum]]['href']
            # check 's3' is at position [:2]
            if getBand.startswith('s3', 0, 2):
                BandList.append(getBand)
        except Exception as e:
            print(e)
                
    #BandList.sort()
    #print(BandList)
    return BandList
    
def GetBandLists(inJSON, bandnum, comp_type):
    
    BandList = []
    with open(inJSON) as f:
        response = json.load(f)
        
    for i in range(len(response['features'])):
        
        if comp_type=='HLS':
            # Get the HLS product type and the product-specific bands from each feature of the JSON
            product_type = response['features'][i]['id'].split('.')[1]
        elif comp_type=='LC2SR':
            # Get the Landsat product type and the product-specific bands from each feature of the JSON
            product_type = response['features'][i]['id'].split('_')[0]
        else:
            print(f"comp type ({comp_type}) not recognized")
            os._exit(1)

        if product_type=='L30':
            bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
        elif product_type=='S30':
            bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B8A', 6:'B11', 7:'B12',8:'Fmask'})
        ## LC2SR ##
        # TODO: configure cloud mask in CreateNDVIstack_LC2SR() using cloud_qa
        elif product_type=='LT04':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'cloud_qa'})
        elif product_type=='LT05':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'cloud_qa'})
        elif product_type=='LE07':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'cloud_qa'})
        elif product_type=='LC08':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'qa_pixel'})
        elif product_type=='LC09':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'qa_aerosol'})
        else:
            print("HLS product type not recognized: Must be L30 or S30.")
            os._exit(1)
            
        try:
            #print(f'GetBandLists: {comp_type} {product_type} {bands[bandnum]}')
            getBand = response['features'][i]['assets'][bands[bandnum]]['href']
            # check 's3' is at position [:2]
            if getBand.startswith('s3', 0, 2):
                BandList.append(getBand)
        except Exception as e:
            print(e)
                
    return BandList

def HLS_MASK(ma_fmask, 
             MASK_LIST=['cloud', 'adj_cloud', 'cloud shadow', 'snowice', 'water', 'aerosol_high'], 
             HLS_QA_BIT = {'cirrus': 0, 'cloud': 1, 'adj_cloud': 2, 'cloud shadow':3, 'snowice':4, 'water':5, 'aerosol_l': 6, 'aerosol_h': 7}):
    
    # This function takes the HLS Fmask layer as a masked array and exports the desired mask image array. 
    # The mask_list assigns the QA conditions you would like to mask.
    # The default mask_list setting is coded for a vegetation application, so it also removes water and snow/ice.
    # See HLS user guide for more details: https://lpdaac.usgs.gov/documents/1326/HLS_User_Guide_V2.pdf
    
    arr = ma_fmask.data
    msk = np.zeros_like(arr)#.astype(np.bool)
    for m in MASK_LIST:
        if m in HLS_QA_BIT.keys():
            msk += (arr & (1 << HLS_QA_BIT[m]) ) #<--added parantheses 
        if m == 'aerosol_high':
            msk += (arr & (1 << HLS_QA_BIT['aerosol_h']) & (1 << HLS_QA_BIT['aerosol_l']))
        if m == 'aerosol_moderate':
            msk += (arr & (1 << HLS_QA_BIT['aerosol_h']) & (0 << HLS_QA_BIT['aerosol_l']))
        if m == 'aerosol_low':
            msk += (arr & (0 << HLS_QA_BIT['aerosol_h']) & (1 << HLS_QA_BIT['aerosol_l']))
            
    #ma_fmask.mask *= msk > 0 # With *, this will be the intersection of the various bit masks
    ma_fmask.mask += msk > 0 # With +, this will be the union of the various bit masks
    return ma_fmask

def LC2SR_MASK(ma_cloudqa, 
             MASK_LIST=['cloud', 'adj_cloud', 'cloud shadow', 'snow', 'water'], 
             LC2SR_QA_BIT = {'dark dense veg': 0, 'cloud': 1, 'cloud shadow': 2, 'adj_cloud':3, 'snow':4, 'water':5}):
    
    # This function takes the LC2SR CLOUD_QA layer as a masked array and exports the desired mask image array. 
    # The mask_list assigns the QA conditions you would like to mask.
    # The default mask_list setting is coded for a vegetation application, so it also removes water and snow.
    # See LC2SR user guide for more details: <TODO: get user guide>
    
    arr = ma_cloudqa.data
    msk = np.zeros_like(arr)#.astype(np.bool)
    for m in MASK_LIST:
        if m in LC2SR_QA_BIT.keys():
            msk += (arr & (1 << LC2SR_QA_BIT[m]) ) #<--added parantheses 
            
    #ma_fmask.mask *= msk > 0 # With *, this will be the intersection of the various bit masks
    ma_cloudqa.mask += msk > 0 # With +, this will be the union of the various bit masks
    return ma_cloudqa

def MaskArrays(file, in_bbox, height, width, comp_type, epsg="epsg:4326", dst_crs="epsg:4326", incl_trans=False, do_mask=False):
    '''Read a window of data from the raster matching the tile bbox
    Return a masked array for the window (subset) of the input file
    or
    Return the image crs and transform (incl_trans=True).
    Note: could be renamed to Get_MaskArray_Subset() '''
    
    with COGReader(file) as cog:
        img = cog.part(in_bbox, bounds_crs=epsg, max_size=None, dst_crs=dst_crs, height=height, width=width)
    if incl_trans:
        return img.crs, img.transform
    
    if comp_type=="HLS":
        if do_mask:
            # Returns the integer Fmask whose bits can be converted to a datamask
            return (np.squeeze(img.array.astype(int)) )
        else:
            return (np.squeeze(img.array.astype(np.float32)) * 0.0001)
        
    elif comp_type=="LC2SR":
        if do_mask:
            # Returns the integer Fmask whose bits can be converted to a datamask
            return (np.squeeze(img.array.astype(int)) )
        else:
            # Surface reflectance collection 2 scaling offset (0.0000275) and bias (- 0.2)
            return (np.squeeze(img.array.astype(np.float32)) * 0.0000275) - 0.2
    else:
        print("composite type not recognized")
        os._exit(1)

def CreateNDVIstack_HLS(REDfile, NIRfile, fmask, in_bbox, epsg, dst_crs, height, width, comp_type, rangelims_red = [0.01, 0.1]):
    '''Calculate NDVI for each source scene
    Mask out pixels above or below the red band reflectance range limit values'''
    
    NIRarr = MaskArrays(NIRfile, in_bbox, height, width, comp_type, epsg, dst_crs)
    REDarr = MaskArrays(REDfile, in_bbox, height, width, comp_type, epsg, dst_crs)
    fmaskarr = MaskArrays(fmask, in_bbox, height, width, comp_type, epsg, dst_crs, do_mask=True)
    
    #
    # HLS masking
    #
    fmaskarr = HLS_MASK(fmaskarr)
    
    #print(f'printing fmaskarr data:\n{fmaskarr.data}')
    #print(f'printing fmaskarr mask:\n{fmaskarr.mask}')
    #ndvi = np.ma.array((NIRarr-REDarr)/(NIRarr+REDarr))
    #print(ndvi.shape)
    
    print(f'Min, max Red value before mask: {REDarr.min()}, {REDarr.max()}')
    print(f'Red rangelims: {rangelims_red}')
    return np.ma.array(np.where(((fmaskarr==1) | (REDarr < rangelims_red[0]) | (REDarr > rangelims_red[1])), -9999, (NIRarr-REDarr)/(NIRarr+REDarr)))
    
def CreateNDVIstack_LC2SR(REDfile, NIRfile, fmask, in_bbox, epsg, dst_crs, height, width, comp_type, rangelims_red = [0.01, 0.1]):
    '''Calculate NDVI for each source scene'''
    NIRarr = MaskArrays(NIRfile, in_bbox, height, width, comp_type, epsg, dst_crs)
    REDarr = MaskArrays(REDfile, in_bbox, height, width, comp_type, epsg, dst_crs)
    fmaskarr = MaskArrays(fmask, in_bbox, height, width, comp_type, epsg, dst_crs, do_mask=True)
    
    #
    # LC2SR masking
    #
    fmaskarr = LC2SR_MASK(fmaskarr)
    
    #return np.ma.array((NIRarr-REDarr)/(NIRarr+REDarr))
    return np.ma.array(np.where(((fmaskarr==1) | (REDarr < rangelims_red[0]) | (REDarr > rangelims_red[1])), -9999, (NIRarr-REDarr)/(NIRarr+REDarr)))

def CollapseBands(inArr, NDVItmp, BoolMask):
    '''
    Inserts the bands as arrays (made earlier)
    Creates a single layer by using the binary mask and a sum function to collapse n-dims to 2-dims
    '''
    inArr = np.ma.masked_equal(inArr, 0)
    inArr[np.logical_not(NDVItmp)]=0 
    compImg = np.ma.masked_array(inArr.sum(0), BoolMask)
    #print(compImg)
    return compImg

def CreateComposite(file_list, NDVItmp, BoolMask, in_bbox, height, width, epsg, dst_crs, comp_type):
    print("\t\tMaskedFile")
    MaskedFile = [MaskArrays(file_list[i], in_bbox, height, width, comp_type, epsg, dst_crs) for i in range(len(file_list))]
    print("\t\tComposite")
    Composite=CollapseBands(MaskedFile, NDVItmp, BoolMask)
    return Composite

def createJulianDateLC2SR(file, height, width):
    date_string = file.split('/')[-1].split('_')[3]
    fmt = '%Y.%m.%d'
    date = date_string[:4] + '.' + date_string[4:6] + '.' + date_string[6:]
    dt = datetime.datetime.strptime(date, fmt)
    tt = dt.timetuple()
    jd = tt.tm_yday
    date_arr = np.full((height, width), jd,dtype=np.float32)
    return date_arr
    
def JulianCompositeLC2SR(file_list, NDVItmp, BoolMask, height, width):
    JulianDateImages = [createJulianDateLC2SR(file_list[i], height, width) for i in range(len(file_list))]
    JulianComposite = CollapseBands(JulianDateImages, NDVItmp, BoolMask)
    return JulianComposite

def createJulianDateHLS(file, height, width):
    j_date = file.split('/')[-1].split('.')[3][4:7]
    date_arr = np.full((height, width),j_date,dtype=np.float32)
    return date_arr
    
def JulianCompositeHLS(file_list, NDVItmp, BoolMask, height, width):
    JulianDateImages = [createJulianDateHLS(file_list[i], height, width) for i in range(len(file_list))]
    JulianComposite = CollapseBands(JulianDateImages, NDVItmp, BoolMask)
    return JulianComposite

def JulianComposite(file_list, NDVItmp, BoolMask, height, width, comp_type):
    if comp_type == 'LC2SR':
        JulianDateImages = [createJulianDateLC2SR(file_list[i], height, width) for i in range(len(file_list))]
    elif comp_type == 'HLS':
        JulianDateImages = [createJulianDateHLS(file_list[i], height, width) for i in range(len(file_list))]
    JulianComposite = CollapseBands(JulianDateImages, NDVItmp, BoolMask)
    return JulianComposite

def year_band(file, height, width, comp_type):
    if comp_type == "HLS":
        year = file.split('/')[-1].split('.')[3][0:4]
    elif comp_type == "LC2SR":
        year = file.split('/')[-1].split('_')[3][0:4]
        
    year_arr = np.full((height, width),year,dtype=np.float32)
    
    return year_arr

def year_band_composite(file_list, NDVItmp, BoolMask, height, width, comp_type):
    year_imgs = [year_band(file_list[i], height, width, comp_type) for i in range(len(file_list))]
    year_composite = CollapseBands(year_imgs, NDVItmp, BoolMask)
    return year_composite

# Vegetation Indices Calculations
# https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices

# SAVI
def calcSAVI(red, nir):
    savi = ((nir - red)/(nir + red + 0.5))*(1.5)
    print('\tSAVI Created')
    return savi

# MSAVI
def calcMSAVI(red, nir):
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    print('\tMSAVI Created')
    return msavi

# NDMI
def calcNDMI(nir, swir):
    ndmi = (nir - swir)/(nir + swir)
    print('\tNDMI Created')
    return ndmi

# EVI
def calcEVI(blue, red, nir):
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    print('\tEVI Created')
    return evi

# NBR
def calcNBR(nir, swir2):
    nbr = (nir - swir2)/(nir + swir2)
    print('\tNBR Created')
    return nbr

# NBR2
def calcNBR2(swir, swir2):    
    nbr2 = (swir - swir2)/(swir + swir2)
    print('\tNBR2 Created')
    return nbr2

def tasseled_cap(bands):
    '''
    Compute the tasseled cap indices: brightness, greenness, wetness
    bands - a 6-layer 3-D (images) or 2-D array (samples) or an OrderedDict with appropriate band names
    tc_coef - a list of 3 tuples, each with 6 coefficients
    '''
    # Tasseled Cap (At Satellite)
    #tc_coef = [
    #(0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872), #brightness
    #(-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608), #greenness
    #(0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559) #wetness
    #]
    
    # Tasseled Cap (SREF: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147121#sec028)
    tc_coef = [
    (0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303), #brightness
    (-0.1603, 0.2819, -0.4934, 0.7940, -0.0002, -0.1446), #greenness
    (0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109) #wetness
    ]
    
    tc = np.zeros((len(np.shape(bands)), bands.shape[1], bands.shape[2]), dtype = np.float32())
    
    #print(np.max(in_arr_sref))
    for i, t in enumerate(tc_coef):
        for b in range(5): # should be 6
            tc[i] += (bands[b] * t[b]).astype(np.float32())
           
    print('\tTassCap Created')
    return tc[0], tc[1], tc[2] 

    # TC Code adapted from: https://github.com/bendv/waffls/blob/master/waffls/indices.py
    # TC coeffs from: https://doi.org/10.1080/2150704X.2014.915434 (OLD at satellite coeffs)
    # New coeffs are in sup table 2 here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147121#sec028
    # which in turn are from/built from: Crist 1985. These are sensor non-specific so should be applicable
    # irrespective of sensor and collection, provided it is SREF

def VegMask(NDVI, MIN_NDVI = 0.1):
    print(f'Min NDVI value before mask: {np.nanmin(np.where(NDVI == -9999, np.nan, NDVI))}')
    print(f'Min NDVI threshold: {MIN_NDVI}')
    mask = np.zeros_like(NDVI)
    mask = np.where(NDVI > MIN_NDVI, 1, mask)
    print(f"\tVegetation mask created: valid data where NDVI > {MIN_NDVI}")
    return mask

def get_pixel_coords(arr, transform):
    rows = np.arange(0,np.shape(arr)[0],1)
    Yarr = (((transform[2]+(0.5*transform[0])) + (rows*transform[0])).reshape(np.shape(arr)[0],1))[::-1]
    Ygeo = np.tile(Yarr, np.shape(arr)[1]).astype(np.float32())
    cols = np.arange(0,np.shape(arr)[1],1)
    Xarr = ((transform[5]+(0.5*transform[4])) + (cols*transform[4]))[::-1]
    Xgeo = np.tile(Xarr, (np.shape(arr)[0],1)).astype(np.float32())
    
    return Xgeo, Ygeo

def renew_session(comp_type):
    if comp_type == 'HLS':
        aws_session = get_aws_session_DAAC(get_creds_DAAC())
    elif comp_type == 'LC2SR':
        aws_session = get_aws_session(get_creds())
    return aws_session   

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_tile_fn", type=str, default="/projects/shared-buckets/nathanmthomas/boreal_tiles_v003.gpkg", help="The filename of the stack's set of vector tiles")
    parser.add_argument("-n", "--in_tile_num", type=int, help="The id of a tile that will define the bounds of the raster stacking")
    parser.add_argument("-o", "--output_dir", type=str, help="The path for the JSON files to be written")
    parser.add_argument("-b", "--tile_buffer_m", type=float, default=0, help="The buffer size (m) applied to the extent of the specified stack tile")
    parser.add_argument("-r", "--res", type=int, default=30, help="The output resolution of the stack")
    parser.add_argument("--shape", type=int, default=None, help="The output height and width of the grid's shape. If None, get from input tile.")    
    parser.add_argument("-lyr", "--in_tile_layer", type=str, default=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-in_tile_id_col", type=str, default="tile_num", help="The column of the tile layer name of the stack tiles dataset that holds the tile num")
    parser.add_argument("-a", "--sat_api", type=str, default="https://landsatlook.usgs.gov/sat-api", help="URL of USGS query endpoint")
    parser.add_argument("-j", "--json_file", type=str, default=None, help="The S3 path to the query response json")
    parser.add_argument("-l", "--local", type=bool, default=False, help="Dictate whether it is a run using local paths")
    parser.add_argument("-sy", "--start_year", type=str, default="2020", help="specify the start year date (e.g., 2020)")
    parser.add_argument("-ey", "--end_year", type=str, default="2021", help="specify the end year date (e.g., 2021)")
    parser.add_argument("-smd", "--start_month_day", type=str, default="06-01", help="specify the start month and day (e.g., 06-01)")
    parser.add_argument("-emd", "--end_month_day", type=str, default="09-15", help="specify the end month and day (e.g., 09-15)")
    parser.add_argument("-mc", "--max_cloud", type=int, default=40, help="specify the max amount of cloud")
    parser.add_argument("-t", "--composite_type", choices=['HLS','LC2SR'], nargs="?", type=str, default='HLS', const='HLS', help="Specify the composite type")
    parser.add_argument("--rangelims_red", type=float, nargs=2, action='store', default=[0.01, 0.1], help="The range limits for red reflectance outside of which will be masked out")
    parser.add_argument("-hls", "--hls_product", choices=['S30','L30','H30'], nargs="?", type=str, default='L30', help="Specify the HLS product; M30 is our name for a combined HLS composite")
    parser.add_argument("-hlsv", "--hls_product_version", type=str, default='2.0', help="Specify the HLS product version")
    parser.add_argument("-ndvi", "--thresh_min_ndvi", type=float, default=0.1, help="NDVI threshold above which vegetation is valid.")
    parser.add_argument("-min_n", "--min_n_filt_results", type=int, default=0, help="Min number of filtered search results desired before hitting max cloud limit.")
    parser.add_argument('--search_only', dest='search_only', action='store_true', help='Only perform search and return response json. No composites made.')
    parser.set_defaults(search_only=False)
    #parser.add_argument("-bnames", "--bandnames_list", nargs="+", default='Blue Green Red NIR SWIR SWIR2 NDVI SAVI MSAVI NDMI EVI NBR NBR2 TCB TCG TCW ValidMask Xgeo Ygeo JulianDate yearDate', help="List of bandnames for composite.")
    args = parser.parse_args()    
    
    '''
    Build multi-spectral (ms) composites with scenes from queries of:
    (a) An endpoint of the USGS Landsat-5/7/8 archive
    (b) An endpoint of the v 2.0 of the HLS S30, L30 archive
    
    The ms composite will have the following bands:
    
    'Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIR2': surface reflectance values for ms composite obs.
    'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2' : indices calc'd from the surface reflectance values of the ms composite obs.
    'TCB', 'TCG', 'TCW' : tasseled cap values from the ms composite obs.
    'ValidMask' : a mask identifing valid vegetation obs.
    'Xgeo' : x coordinate
    'Ygeo' : y coordinate
    'JulianDate' : day-of-year of ms composite obs.
    'yearDate' : year of ms composite obs.
    'count': the pixelwise count of valid observations considered during compositing.
    
    Note: 
    HLS info:
    https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/harmonized-landsat-sentinel-2-hls-overview/  
    
    LDPAAC Forum:
    https://forum.earthdata.nasa.gov/viewforum.php?f=7
    
    GitHub CMR issue:
    https://github.com/nasa/cmr-stac/issues
    '''
    
    bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIR2', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'ValidMask', 'Xgeo', 'Ygeo', 'JulianDate', 'yearDate','count']
    
    geojson_path_albers = args.in_tile_fn
    print('\nTiles path:\t\t', geojson_path_albers)
    tile_n = args.in_tile_num
    print("Tile number:\t\t", tile_n)
    res = args.res
    print("Output res (m):\t\t", res)
    
    tile_buffer_m = args.tile_buffer_m
    
    # Get tile by number form GPKG. Store box and out crs
    tile_id = get_index_tile(vector_path=geojson_path_albers, id_col=args.in_tile_id_col, tile_id=tile_n, buffer=tile_buffer_m, layer = args.in_tile_layer)#layer = "boreal_tiles_albers"
    #in_bbox = tile_id['bbox_4326']
    in_bbox = tile_id['geom_orig_buffered'].bounds.iloc[0].to_list()
    out_crs = tile_id['tile_crs']
    #print(out_crs)
    
    print("in_bbox:\t\t", in_bbox)
    print('bbox 4326:\t\t', tile_id['bbox_4326'])
    #print("out_crs = ", out_crs)

    # This is added to allow the output size to be forced to a certain size - this avoids have some tiles returned as 2999 x 3000 due to rounding issues.
    # Most tiles dont have this problem and thus dont need this forced shape, but some consistently do. 
    if args.shape is None:
        print(f'Getting output dims from buffered (buffer={tile_buffer_m}) original tile geometry...')
        height, width = get_shape(in_bbox, res)
    else:
        print('Getting output dims from input shape arg...')
        height = args.shape
        width = args.shape
    
    print(f'Output dims:\t\t{height} x {width}')
    
    # specify either -j or -a. 
    # -j is the path to the ready made json files with links to LS buckets (needs -o to placethe composite)
    # -a is the link to the api to search for data (needs -o to fetch the json files from at_api)
    
    print(f'Composite type:\t\t{args.composite_type}')
    
    # TODO: Change this section to be able to read JSON files from S3
    if args.json_file == None:
        if args.output_dir == None:
            print("MUST SPECIFY -o FOR JSON PATH")
            os._exit(1)
        elif args.composite_type == 'HLS':
            ms_product = args.hls_product
            ms_version = args.hls_product_version
            # master_json = get_HLS_data(args.in_tile_fn, args.in_tile_layer, args.in_tile_id_col, args.in_tile_num, args.output_dir, args.sat_api, 
            #                            args.start_year, args.end_year, args.start_month_day, args.end_month_day, args.max_cloud, args.local, 
            #                            ms_product, ms_version, args.min_n_filt_results, bands_dict=MS_BANDS_DICT)
        elif args.composite_type == 'LC2SR':
            # master_json = get_ls8_data(args.in_tile_fn, args.in_tile_layer, args.in_tile_id_col, args.in_tile_num, args.output_dir, args.sat_api, 
            #                            args.start_year, args.end_year, args.start_month_day, args.end_month_day, args.max_cloud, args.local)
            # master_json = get_LC2SR_data(args.in_tile_fn, args.in_tile_layer, args.in_tile_id_col, args.in_tile_num, args.output_dir, args.sat_api, 
            #                            args.start_year, args.end_year, args.start_month_day, args.end_month_day, args.max_cloud, args.local, 
            #                            args.min_n_filt_results, bands_dict=MS_BANDS_DICT)
            ms_product = 'landsat-c2l2-sr'
            ms_version = None
        else:
            print("specify the composite type (HLS, LC2SR)")
            os._exit(1)
            
        # Get the multispectral data for compositing as a json file of CMR query responses
        master_json = get_ms_data(args.in_tile_fn, args.in_tile_layer, args.in_tile_id_col, args.in_tile_num, args.output_dir, args.sat_api, 
                                args.start_year, args.end_year, args.start_month_day, args.end_month_day, args.max_cloud, 
                                args.composite_type, args.local, ms_product, ms_version, args.min_n_filt_results, bands_dict=MS_BANDS_DICT)
    else:
        master_json = args.json_file
    
    if args.search_only:
        print(f"Search only mode. Master JSON written: {master_json}")
        os._exit(1)
    
    blue_bands = GetBandLists(master_json, 2, args.composite_type)#, product_type=args.hls_product)
    print(f"\nTotal # of scenes for composite:\t\t{len(blue_bands)}")
    if len(blue_bands) == 0:
            print("\nNo scenes to build a composite. Exiting.\n")
            os._exit(1)  
    #print(f"Example path to a band:\t\t {blue_bands[0]}")
    green_bands = GetBandLists(master_json, 3, args.composite_type)
    red_bands =   GetBandLists(master_json, 4, args.composite_type)
    nir_bands =   GetBandLists(master_json, 5, args.composite_type)
    swir_bands =  GetBandLists(master_json, 6, args.composite_type)
    swir2_bands = GetBandLists(master_json, 7, args.composite_type)
    fmask_bands = GetBandLists(master_json, 8, args.composite_type)
  
    ## Create NDVI layers
    ## Loops over lists of bands and calculates NDVI
    ## creates a new list of NDVI images, one per input scene
    print('Creating NDVI stack...')
    print(args.composite_type)
    # insert AWS credentials here if needed
    if args.composite_type == 'HLS':
        aws_session = get_aws_session_DAAC(get_creds_DAAC())
    elif args.composite_type == 'LC2SR':
        aws_session = get_aws_session(get_creds())
    else:
        print("specify the composite type (HLS, LC2SR)")
        os._exit(1)
    
    #print(aws_session)
    
    # Start reading data: TO DO: nmt28 - edit so it can use comp_type
    with rio.Env(aws_session):
        in_crs, crs_transform = MaskArrays(red_bands[0], in_bbox, height, width, args.composite_type, out_crs, out_crs, incl_trans=True)
        if args.composite_type=='HLS':
            NDVIstack = [CreateNDVIstack_HLS(red_bands[i],nir_bands[i],fmask_bands[i], in_bbox, out_crs, out_crs, height, width, args.composite_type, rangelims_red = args.rangelims_red) for i in range(len(red_bands))]
        elif args.composite_type=='LC2SR':
            NDVIstack = [CreateNDVIstack_LC2SR(red_bands[i],nir_bands[i],fmask_bands[i], in_bbox, out_crs, out_crs, height, width, args.composite_type, rangelims_red = args.rangelims_red) for i in range(len(red_bands))]
        
        print('finished')
       
    # Create Bool mask where there is no value in any of the NDVI layers
    print("Make NDVI masked array")
    NDVIstack_ma = np.ma.array(NDVIstack)
    print("shape:\t\t", NDVIstack_ma.shape)
    MaxNDVI = np.ma.max(NDVIstack_ma,axis=0)
    BoolMask = np.ma.getmask(MaxNDVI)
    del MaxNDVI
    
    ## Get the argmax index positions from the stack of NDVI images
    print("Calculate Stack max NDVI image")
    NDVImax = np.nanargmax(NDVIstack_ma, axis=0)
    ## create a tmp array (binary mask) of the same input shape
    NDVItmp = np.ma.zeros(NDVIstack_ma.shape, dtype=bool)
    
    ## Get the pixelwise count of the valid data
    CountComp = np.sum((NDVIstack_ma != -9999), axis=0)
    print(f"Count array min ({CountComp.min()}), max ({CountComp.max()}), and shape ({CountComp.shape})")
    
    ## for each dimension assign the index position (flattens the array to a LUT)
    print("Create LUT of max NDVI positions")
    for i in range(np.shape(NDVIstack_ma)[0]):
        NDVItmp[i,:,:]=NDVImax==i
      
    # create band-by-band composites: TODO multiprocess these
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Blue Composite')
        BlueComp = CreateComposite(blue_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Green Composite')
        GreenComp = CreateComposite(green_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Red Composite')
        RedComp = CreateComposite(red_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating NIR Composite')
        NIRComp = CreateComposite(nir_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating SWIR Composite')
        SWIRComp = CreateComposite(swir_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating SWIR2 Composite')
        SWIR2Comp = CreateComposite(swir2_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating NDVI Composite')
        NDVIComp = CollapseBands(NDVIstack_ma, NDVItmp, BoolMask)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session): 
        print('Creating Julian Date Comp')
        # if args.composite_type == 'HLS':  
        #     JULIANcomp = JulianCompositeHLS(swir2_bands, NDVItmp, BoolMask, height, width)
        # elif args.composite_type == 'LC2SR':
        #     JULIANcomp = JulianCompositeLC2SR(swir2_bands, NDVItmp, BoolMask, height, width)
        JULIANcomp = JulianComposite(swir2_bands, NDVItmp, BoolMask, height, width, args.composite_type)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Year Date Comp')
        YEARcomp = year_band_composite(swir2_bands, NDVItmp, BoolMask, height, width, args.composite_type)
    
    print("Generating covariates")
    SAVI =  calcSAVI(RedComp, NIRComp)
    MSAVI = calcMSAVI(RedComp, NIRComp)
    NDMI =  calcNDMI(NIRComp, SWIRComp)
    EVI =   calcEVI(BlueComp, RedComp, NIRComp)
    NBR =   calcNBR(NIRComp, SWIR2Comp)
    NBR2 =  calcNBR2(SWIRComp, SWIR2Comp)
    TCB, TCG, TCW = tasseled_cap(np.transpose([BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, SWIR2Comp], [0, 1, 2]))
    
    print("Calculating X and Y pixel center coords...")
    ValidMask = VegMask(NDVIComp, MIN_NDVI=args.thresh_min_ndvi)
    Xgeo, Ygeo = get_pixel_coords(ValidMask, crs_transform)
    
    # Stack bands together
    print("\nCreating raster stack...\n")
    # These must correspond with the bandnames
    stack = np.transpose([BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, SWIR2Comp, NDVIComp, SAVI, MSAVI, NDMI, EVI, NBR, NBR2, TCB, TCG, TCW, ValidMask, Xgeo, Ygeo, JULIANcomp, YEARcomp, CountComp], [0, 1, 2]) 
     
    print(f"Assigning band names:\n\t{bandnames}\n")
    print("specifying output directory and filename")

    outdir = args.output_dir
    start_season = args.start_month_day[0:2] + args.start_month_day[2:]
    end_season = args.end_month_day[0:2] + args.end_month_day[2:]
    start_year = args.start_year
    end_year = args.end_year
    comp_type = args.composite_type
    out_stack_fn = os.path.join(outdir, comp_type + '_' + str(tile_n) + '_' + start_season + '_' + end_season + '_' + start_year + '_' + end_year + '.tif')
    
    if False:
        #'''Backfilling needs testing...'''
        #from 3.1.2_gap_fill_dps import *
        fill_stack = build_backfill_composite_HLS(in_tile_fn, in_tile_num, resolution, tile_buffer_m, in_tile_layer, out_dir, comp_type, sat_api, \
                                                  start_year-1, start_year-1, start_month_day, end_month_day, \
                                                  max_cloud, local_json=None)
        stack = np.where(stack==-9999, fill_stack, stack)
    if True:
        print('\nApply a common mask across all layers of stack...')
        print(f"Stack shape pre-mask:\t\t{stack.shape}")
        stack[:, np.any(stack == -9999, axis=0)] = -9999 
        print(f"Stack shape post-mask:\t\t{stack.shape}")

    # write COG to disk
    write_cog(stack, 
              out_stack_fn, 
              in_crs, 
              crs_transform, 
              bandnames, 
              out_crs=out_crs, 
              resolution=(res, res), 
              align=True, ### Debug
              clip_geom=tile_id["geom_orig"] # this was added late to address some HLS output showing 2999 rows..now this matches how topo stacks are built. Does not correct issue.
             )
    print(f"Wrote out stack:\t\t{out_stack_fn}\n")
    return(out_stack_fn)
    
if __name__ == "__main__":
    '''
    
    if args.back_fill == True:
        start_year = 2019
        end_year = 2019
        start_month_day = '06-01'
        end_month_day = '09-15'
        max_cloud = 40
        
        build_backfill_composite_HLS(geojson_path_albers, tile_n, res, args.tile_buffer_m, args.in_tile_layer, outdir, comp_type, sat_api, start_year, end_year, start_month_day, end_month_day, max_cloud, args.local)
    
    '''
    main()
    
    

