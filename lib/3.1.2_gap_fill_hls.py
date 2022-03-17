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
from CovariateUtils import write_cog, get_index_tile, get_aws_session, get_aws_session_DAAC
from fetch_HLS import get_HLS_data
from fetch_from_api import get_ls8_data
import json
import datetime
from maap.maap import MAAP
maap = MAAP(maap_host='api.ops.maap-project.org')

def get_shape(bbox, res=30):
    left, bottom, right, top = bbox
    width = int((right-left)/res)
    height = int((top-bottom)/res)
    return height,width

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

def GetBandLists(inJSON, bandnum, comp_type):
    if comp_type=='HLS':
        bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
    elif comp_type=='LS8':
        bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22'})
    else:
        print("comp type not recognized")
        os._exit(1)
    #bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
    BandList = []
    with open(inJSON) as f:
        response = json.load(f)
    for i in range(len(response['features'])):
        try:
            getBand = response['features'][i]['assets'][bands[bandnum]]['href']
            # check 's3' is at position [:2]
            if getBand.startswith('s3', 0, 2):
                BandList.append(getBand)
        except Exception as e:
            print(e)
                
    BandList.sort()
    return BandList

def MaskArrays(file, in_bbox, height, width, comp_type, epsg="epsg:4326", dst_crs="epsg:4326", incl_trans=False):
    '''Read a window of data from the raster matching the tile bbox'''
    #print(file)
    
    with COGReader(file) as cog:
        img = cog.part(in_bbox, bounds_crs=epsg, max_size=None, dst_crs=dst_crs, height=height, width=width)
    if incl_trans:
        return img.crs, img.transform
    # Surface reflectance collection 2 scaling offset and bias
    # 0.0000275 + -0.2
    
    if comp_type=="HLS":
        print("HLS")
        return (np.squeeze(img.as_masked().astype(np.float32)) * 0.001)
    elif comp_type=="LS8":
        return (np.squeeze(img.as_masked().astype(np.float32)) * 0.0000275) - 0.2
    else:
        print("composite type not recognized")
        os._exit(1)

def CreateNDVIstack_HLS(REDfile, NIRfile, fmask, in_bbox, epsg, dst_crs, height, width, comp_type):
    '''Calculate NDVI for each source scene'''
    NIRarr = MaskArrays(NIRfile, in_bbox, height, width, comp_type, epsg, dst_crs)
    REDarr = MaskArrays(REDfile, in_bbox, height, width, comp_type, epsg, dst_crs)
    fmaskarr = MaskArrays(fmask, in_bbox, height, width, comp_type, epsg, dst_crs)
    #ndvi = np.ma.array((NIRarr-REDarr)/(NIRarr+REDarr))
    #print(ndvi.shape)
    return np.ma.array(np.where(((fmaskarr==1) | (REDarr>1.0) | (REDarr<0.1)), -9999, (NIRarr-REDarr)/(NIRarr+REDarr)))

def CreateNDVIstack_LS8(REDfile, NIRfile, in_bbox, epsg, dst_crs, height, width, comp_type):
    '''Calculate NDVI for each source scene'''
    NIRarr = MaskArrays(NIRfile, in_bbox, height, width, comp_type, epsg, dst_crs)
    REDarr = MaskArrays(REDfile, in_bbox, height, width, comp_type, epsg, dst_crs)
    #ndvi = np.ma.array((NIRarr-REDarr)/(NIRarr+REDarr))
    #print(ndvi.shape)
    return np.ma.array((NIRarr-REDarr)/(NIRarr+REDarr))

# insert the bands as arrays (made earlier)
## creates a single layer by using the binary mask
## and a sum function to collapse n-dims to 2-dims
def CollapseBands(inArr, NDVItmp, BoolMask):
    inArr = np.ma.masked_equal(inArr, 0)
    inArr[np.logical_not(NDVItmp)]=0 
    compImg = np.ma.masked_array(inArr.sum(0), BoolMask)
    print(compImg)
    return compImg

def CreateComposite(file_list, NDVItmp, BoolMask, in_bbox, height, width, epsg, dst_crs, comp_type):
    print("MaskedFile")
    MaskedFile = [MaskArrays(file_list[i], in_bbox, height, width, comp_type, epsg, dst_crs) for i in range(len(file_list))]
    print("Composite")
    Composite=CollapseBands(MaskedFile, NDVItmp, BoolMask)
    return Composite

def createJulianDate(file, height, width):
    date_string = file.split('/')[-1].split('_')[3]
    fmt = '%Y.%m.%d'
    date = date_string[:4] + '.' + date_string[4:6] + '.' + date_string[6:]
    dt = datetime.datetime.strptime(date, fmt)
    tt = dt.timetuple()
    jd = tt.tm_yday
    
    date_arr = np.full((height, width), jd,dtype=np.float32)
    return date_arr
    
def JulianComposite(file_list, NDVItmp, BoolMask, height, width):
    JulianDateImages = [createJulianDate(file_list[i], height, width) for i in range(len(file_list))]
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

def year_band(file, height, width, comp_type):
    if comp_type == "HLS":
        year = file.split('/')[-1].split('.')[3][0:4]
    elif comp_type == "LS8":
        year = file.split('/')[-1].split('_')[3][0:4]
        
    year_arr = np.full((height, width),year,dtype=np.float32)
    
    return year_arr

def year_band_composite(file_list, NDVItmp, BoolMask, height, width, comp_type):
    year_imgs = [year_band(file_list[i], height, width, comp_type) for i in range(len(file_list))]
    year_composite = CollapseBands(year_imgs, NDVItmp, BoolMask)
    
    return year_composite

# Co-var functions
# Reads in bands on the fly, as needed

# SAVI
def calcSAVI(red, nir):
    savi = ((nir - red)/(nir + red + 0.5))*(1.5)
    print('SAVI Created')
    return savi

# MASAVI
def calcMSAVI(red, nir):
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    print('MSAVI Created')
    return msavi

# NDMI
def calcNDMI(nir, swir):
    ndmi = (nir - swir)/(nir + swir)
    print('NDMI Created')
    return ndmi

# EVI
def calcEVI(blue, red, nir):
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    print('EVI Created')
    return evi

# NBR
def calcNBR(nir, swir2):
    nbr = (nir - swir2)/(nir + swir2)
    print('NBR Created')
    return nbr

# NBR2
def calcNBR2(swir, swir2):    
    nbr2 = (swir - swir2)/(swir + swir2)
    print('NBR2 Created')
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
           
    print('TassCap Created')
    return tc[0], tc[1], tc[2] 

    # TC Code adapted from: https://github.com/bendv/waffls/blob/master/waffls/indices.py
    # TC coeffs from: https://doi.org/10.1080/2150704X.2014.915434 (OLD at satellite coeffs)
    # New coeffs are in sup table 2 here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147121#sec028
    # which in turn are from/built from: Crist 1985. These are sensor non-specific so should be applicable
    # irrespective of sensor and collection, provided it is SREF

def VegMask(NDVI):
    mask = np.zeros_like(NDVI)
    mask = np.where(NDVI > 0.1, 1, mask)
    print("Veg Mask Created")
    return mask

def get_pixel_coords(arr, transform):
    rows = np.arange(0,np.shape(arr)[0],1)
    Yarr = (((transform[2]+(0.5*transform[0])) + (rows*transform[0])).reshape(np.shape(arr)[0],1))[::-1]
    Ygeo = np.tile(Yarr, np.shape(arr)[1]).astype(np.float32())
    cols = np.arange(0,np.shape(arr)[1],1)
    Xarr = ((transform[5]+(0.5*transform[4])) + (cols*transform[4]))[::-1]
    Xgeo = np.tile(Xarr, (np.shape(arr)[0],1)).astype(np.float32())
    
    return Xgeo, Ygeo

def build_backfill_composite_HLS(in_tile_fn, in_tile_num, resolution, tile_buffer_m, in_tile_layer, out_dir, comp_type, sat_api, start_year, end_year, start_month_day, end_month_day, max_cloud, local_json=None)

    # EXAMPLE CALL
    # python 3.1.2_dps.py -i /projects/maap-users/alexdevseed/boreal_tiles.gpkg -n 30543 -l boreal_tiles_albers  -o /projects/tmp/Landsat/ -b 0 -a https://landsatlook.usgs.gov/sat-api
    geojson_path_albers = in_tile_fn
    print('geopkg path = ', geojson_path_albers)
    tile_n = in_tile_num
    print("tile number = ", tile_n)
    res = resolution
    print("output resolution = ", res)
    
    # Get tile by number form GPKG. Store box and out crs
    tile_id = get_index_tile(geojson_path_albers, tile_n, tile_buffer_m, layer = in_tile_layer)#layer = "boreal_tiles_albers"
    #in_bbox = tile_id['bbox_4326']
    in_bbox = tile_id['geom_orig_buffered'].bounds.iloc[0].to_list()
    out_crs = tile_id['tile_crs']
    print(out_crs)
    
    print("in_bbox = ", in_bbox)
    print("out_crs = ", out_crs)

    
    height, width = get_shape(in_bbox, res)
    
    
    # specify either -j or -a. 
    # -j is the path to the ready made json files with links to LS buckets (needs -o to placethe composite)
    # -a is the link to the api to search for data (needs -o to fetch the json files from at_api)
    
    # TODO: Change this section to be able to read JSON files from S3
    if local_json == None:
        if out_dir == None:
            print("MUST SPECIFY -o FOR JSON PATH")
            os._exit(1)
        elif comp_type == 'HLS':
            print("get HLS data")
            master_json = get_HLS_data(in_tile_fn, in_tile_layer, in_tile_num, output_dir, sat_api, start_year, end_year, start_month_day, end_month_day, max_cloud, local_json)
        elif composite_type == 'LS8':
            print("get ls8 data")
            master_json = get_ls8_data(in_tile_fn, in_tile_layer, in_tile_num, output_dir, sat_api, start_year, end_year, start_month_day, end_month_day, max_cloud, local_json)
        else:
            print("specify the composite type (HLS, LS8)")
            os._exit(1)
    else:
        print("no json file")
        
    
    
    blue_bands = GetBandLists(master_json, 2, comp_type)
    print("Number of files per band =", len(blue_bands))
    print(blue_bands[0])
    
    green_bands = GetBandLists(master_json, 3, comp_type)
    red_bands = GetBandLists(master_json, 4, comp_type)
    nir_bands = GetBandLists(master_json, 5, comp_type)
    swir_bands = GetBandLists(master_json, 6, comp_type)
    swir2_bands = GetBandLists(master_json, 7, comp_type)
    if comp_type=='HLS':
        fmask_bands = GetBandLists(master_json, 8, comp_type)
    
    print("Number of files per band =", len(blue_bands))
    #print(blue_bands[0])
    
    
    ## create NDVI layers
    ## Loopsover lists of bands and calculates NDVI
    ## creates a new list of NDVI images, one per input scene
    print('Creating NDVI stack...')
    print(comp_type)
    # insert AWS credentials here if needed
    if comp_type == 'HLS':
        aws_session = get_aws_session_DAAC()
    elif comp_type == 'LS8':
        aws_session = get_aws_session()
    else:
        print("specify the composite type (HLS, ls8)")
        os._exit(1)
    
    print(aws_session)
    
    # Start reading data: TO DO: nmt28 - edit so it can use comp_type
    with rio.Env(aws_session):
        in_crs, crs_transform = MaskArrays(red_bands[0], in_bbox, height, width, comp_type, out_crs, out_crs, incl_trans=True)
        print(in_crs)
        if comp_type=='HLS':
            NDVIstack = [CreateNDVIstack_HLS(red_bands[i],nir_bands[i],fmask_bands[i], in_bbox, out_crs, out_crs, height, width, comp_type) for i in range(len(red_bands))]
        elif comp_type=='LS8':
            NDVIstack = [CreateNDVIstack_LS8(red_bands[i],nir_bands[i], in_bbox, out_crs, out_crs, height, width, comp_type) for i in range(len(red_bands))]
        
        print('finished')
    
    
    # Create Bool mask where there is no value in any of the NDVI layers
    print("Make NDVI valid mask")
    print("shape = ", np.ma.array(NDVIstack).shape)
    MaxNDVI = np.ma.max(np.ma.array(NDVIstack),axis=0)
    BoolMask = np.ma.getmask(MaxNDVI)
    del MaxNDVI
    
    ## Get the argmax index positions from the stack of NDVI images
    print("Get stack nan mask")
    NDVIstack = np.ma.array(NDVIstack)
    print("Calculate Stack max NDVI image")
    NDVImax = np.nanargmax(NDVIstack,axis=0)
    ## create a tmp array (binary mask) of the same input shape
    NDVItmp = np.ma.zeros(NDVIstack.shape, dtype=bool)

    ## for each dimension assign the index position (flattens the array to a LUT)
    print("Create LUT of max NDVI positions")
    for i in range(np.shape(NDVIstack)[0]):
        NDVItmp[i,:,:]=NDVImax==i
    
    
    # create band-by-band composites
    with rio.Env(aws_session):
        print('Creating Blue Composite')
        BlueComp = CreateComposite(blue_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, comp_type)
        print('Creating Green Composite')
        GreenComp = CreateComposite(green_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, comp_type)
        print('Creating Red Composite')
        RedComp = CreateComposite(red_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, comp_type)
        print('Creating NIR Composite')
        NIRComp = CreateComposite(nir_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, comp_type)
        print('Creating SWIR Composite')
        SWIRComp = CreateComposite(swir_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, comp_type)
        print('Creating SWIR2 Composite')
        SWIR2Comp = CreateComposite(swir2_bands, NDVItmp, BoolMask, in_bbox, height, width, out_crs, out_crs, comp_type)
        print('Creating NDVI Composite')
        NDVIComp = CollapseBands(NDVIstack, NDVItmp, BoolMask)
        if comp_type == 'HLS':
            print('Creating Julian Date Comp')
            JULIANcomp = JulianCompositeHLS(swir2_bands, NDVItmp, BoolMask, height, width)
        elif comp_type == 'LS8':
            JULIANcomp = JulianComposite(swir2_bands, NDVItmp, BoolMask, height, width)
        YEARComp = year_band_composite(swir2_bands, NDVItmp, BoolMask, height, width, comp_type)
        
    # calculate covars
    print("Generating covariates")
    SAVI = calcSAVI(RedComp, NIRComp)
    print("MSAVI")
    MSAVI = calcMSAVI(RedComp, NIRComp)
    print("NDMI")
    NDMI = calcNDMI(NIRComp, SWIRComp)
    print("EVI")
    EVI = calcEVI(BlueComp, RedComp, NIRComp)
    print("NBR")
    NBR = calcNBR(NIRComp, SWIR2Comp)
    print("NBR2")
    NBR2 = calcNBR2(SWIRComp, SWIR2Comp)
    print("TCB")
    TCB, TCG, TCW = tasseled_cap(np.transpose([BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, SWIR2Comp], [0, 1, 2]))
    print("calculate X and Y picel center coords")
    ValidMask = VegMask(NDVIComp)
    Xgeo, Ygeo = get_pixel_coords(ValidMask, crs_transform)
    
    # Stack bands together
    print("Create raster stack")
    stack = np.transpose([BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, SWIR2Comp, NDVIComp, SAVI, MSAVI, NDMI, EVI, NBR, NBR2, TCB, TCG, TCW, ValidMask, Xgeo, Ygeo, JULIANcomp], [0, 1, 2]) 
    print("Assign band names")
    #assign band names
    bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIR2', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'ValidMask', 'Xgeo', 'Ygeo', 'JulianDate']
    print("specifying output directory and filename")
    #outdir = '/projects/tmp/Landsat'
    outdir = output_dir
    start_season = start_month_day[0:2] + start_month_day[2:]
    end_season = end_month_day[0:2] + end_month_day[2:]
    start_year = start_year
    end_year = end_year
    comp_type = composite_type
    out_stack_fn = os.path.join(outdir, comp_type + '_' + str(tile_n) + '_' + start_season + '_' + end_season + '_' + start_year + '_' + end_year + '.tif')
    
    return(out_stack_fn)
    

    
    

