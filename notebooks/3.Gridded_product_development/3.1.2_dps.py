import argparse
#import requests
#import datetime
#import sys
#import tarfile
import json
import os

import geopandas as gpd
#import shapely as shp
import boto3
import rasterio as rio
import geopandas as gpd
from shapely.geometry import box
#from fiona.crs import from_epsg
from rasterio.mask import mask
from rasterio.warp import *
from rasterio.merge import merge
from rasterio.crs import CRS
#from rasterio import windows
#from rasterio.io import MemoryFile
#from rasterio.transform import from_bounds
#from rasterio.vrt import WarpedVRT
#from rio_cogeo.cogeo import cog_translate
#from rio_cogeo.profiles import cog_profiles
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#import rasterio
#from rasterio.plot import show
from rasterio.session import AWSSession

from CovariateUtils import write_cog, get_index_tile
from fetch_from_api import get_data

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

def GetBandLists(inJSON, bandnum):
    BandList = []
    with open(inJSON) as f:
        response = json.load(f)
    for i in range(len(response['features'])):
        try:
            getBand = response['features'][i]['assets']['SR_B' + str(bandnum) + '.TIF']['href']
            BandList.append(getBand)
        except exception as e:
            print(e)
                
    BandList.sort()
    return BandList

def define_raster(file, in_bbox, epsg="epsg:4326"):
    '''Read the first raster to get its transform and crs'''
    with rio.open(file, 'r') as f:
        bbox = transform_bounds(epsg, f.crs, *in_bbox)
        band, crs_transform = merge([f],bounds=bbox)
    return f.crs, crs_transform

def MaskArrays(file, in_bbox, epsg="epsg:4326"):
    '''Read a window of data from the raster matching the tile bbox'''
    with rio.open(file, 'r') as f:
        bbox = transform_bounds(epsg, f.crs, *in_bbox)
        band, crs_transform = merge([f],bounds=bbox)
    return np.ma.masked_array(band[0].astype(np.float32()), mask=f.nodata)

def CreateNDVIstack(REDfile, NIRfile, in_bbox):
    '''Calculate NDVI for each source scene'''
    NIRarr = MaskArrays(NIRfile, in_bbox)
    REDarr = MaskArrays(REDfile, in_bbox)
    return np.ma.array((NIRarr-REDarr)/(NIRarr+REDarr))

# insert the bands as arrays (made earlier)
## creates a single layer by using the binary mask
## and a sum function to collapse n-dims to 2-dims
def CollapseBands(inArr, NDVItmp, BoolMask):
    inArr = np.ma.masked_equal(inArr, 0)
    inArr[np.logical_not(NDVItmp)]=0 
    compImg = np.ma.masked_array(inArr.sum(0), BoolMask)
    return compImg

def CreateComposite(file_list, NDVItmp, BoolMask, in_bbox):
    MaskedFile = [MaskArrays(file_list[i], in_bbox) for i in range(len(file_list))]
    Composite=CollapseBands(MaskedFile, NDVItmp, BoolMask)

    return Composite

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
    # Tasseled Cap
    tc_coef = [
    (0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872), #brightness
    (-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608), #greenness
    (0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559) #wetness
    ]

    in_arr = bands
    print(np.shape(in_arr))
    tc = np.zeros((len(np.shape(in_arr)), in_arr.shape[1], in_arr.shape[2]), dtype = np.float32())
    
    #print(np.max(in_arr))
    for i, t in enumerate(tc_coef):
        for b in range(5): # should be 6
            tc[i] += (in_arr[b] * t[b]).astype(np.float32())
           
    print('TassCap Created')
    return tc[0], tc[1], tc[2] 

    # TC Code adapted from: https://github.com/bendv/waffls/blob/master/waffls/indices.py
    # TC coefs from: https://doi.org/10.1080/2150704X.2014.915434

def VegMask(NDVI):
    mask = np.zeros_like(NDVI)
    mask = np.where(NDVI > 0.1, 1, mask)
    print("Veg Mask Created")
    return mask

def get_pixel_coords(arr, transform):
    rows = np.arange(0,np.shape(arr)[0],1)
    Ygeo = np.tile(((transform[2]+(0.5*transform[0])) + (rows*transform[0])).reshape(np.shape(arr)[0],1), np.shape(arr)[1]).astype(np.float32())
    cols = np.arange(0,np.shape(arr)[1],1)
    Xgeo = np.tile(((transform[5]+(0.5*transform[4])) + (cols*transform[4])), (np.shape(arr)[0],1)).astype(np.float32())
    
    return Xgeo, Ygeo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_tile_fn", type=str, help="The filename of the stack's set of vector tiles")
    parser.add_argument("-n", "--in_tile_num", type=int, help="The id of a tile that will define the bounds of the raster stacking")
    parser.add_argument("-o", "--output_dir", type=str, help="The path for the JSON files to be written")
    parser.add_argument("-b", "--tile_buffer_m", type=float, default=None, help="The buffer size (m) applied to the extent of the specified stack tile")
    parser.add_argument("-r", "--res", type=int, default=30, help="The output resolution of the stack")
    parser.add_argument("-lyr", "--in_tile_layer", type=str, default=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-a", "--sat_api", type=str, default="https://landsatlook.usgs.gov/sat-api", help="URL of USGS query endpoint")
    parser.add_argument("-j", "--json_file", type=str, default=None, help="The S3 path to the query response json")
    parser.add_argument("-l", "--local", type=bool, default=False, help="Dictate whether it is a run using local paths")
    args = parser.parse_args()    

    # EXAMPLE CALL
    # python 3.1.2_dps.py -i /projects/maap-users/alexdevseed/boreal_tiles.gpkg -n 30543 -l boreal_tiles_albers  -o /projects/tmp/Landsat/ -b 0 -a https://landsatlook.usgs.gov/sat-api
    geojson_path_albers = args.in_tile_fn
    print('geopkg path = ', geojson_path_albers)
    tile_n = args.in_tile_num
    print("tile number = ", tile_n)
    res = args.res
    print("output resolution = ", res)
    
    # Get tile by number form GPKG. Store box and out crs
    tile_id = get_index_tile(geojson_path_albers, tile_n, args.tile_buffer_m, layer = "boreal_tiles_albers",)
    in_bbox = tile_id['bbox_4326']
    out_crs = tile_id['tile_crs']
    
    print("in_bbox = ", in_bbox)
    print("out_crs = ", out_crs)

    # specify either -j or -a. 
    # -j is the path to the ready made json files with links to LS buckets (needs -o to placethe composite)
    # -a is the link to the api to search for data (needs -o to fetch the json files from at_api)
    
    # TODO: Change this section to be able to read JSON files from S3
    if args.json_file == None:
        if args.output_dir == None:
            print("MUST SPECIFY -o FOR JSON PATH")
            os.exit(1)
        else:
            master_json = get_data(args.in_tile_fn, args.in_tile_layer, args.in_tile_num, args.output_dir, args.sat_api, args.local)
    else:
        master_json = args.json_file
        #Download the json from s3
        #inJSON = get_json(master_json)

    #print(master_json)
    
    blue_bands = GetBandLists(master_json, 2)
    green_bands = GetBandLists(master_json, 3)
    red_bands = GetBandLists(master_json, 4)
    nir_bands = GetBandLists(master_json, 5)
    swir_bands = GetBandLists(master_json, 6)
    swir2_bands = GetBandLists(master_json, 7)
    
    print("Number of files per band =", len(blue_bands))
    print(blue_bands[0])
    
    
    ## create NDVI layers
    ## Loopsover lists of bands and calculates NDVI
    ## creates a new list of NDVI images, one per input scene
    print('Creating NDVI stack...')
    # insert AWS credentials here if needed
    aws_session = AWSSession(boto3.Session())
    with rio.Env(aws_session):
        in_crs, crs_transform = define_raster(red_bands[0], in_bbox, epsg="epsg:4326")
        print(in_crs)
        NDVIstack = [CreateNDVIstack(red_bands[i],nir_bands[i],in_bbox) for i in range(len(red_bands))]
        print('finished')
    
    
    # Create Bool mask where there is no value in any of the NDVI layers
    print("Make NDVI valid mask")
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
        BlueComp = CreateComposite(blue_bands, NDVItmp, BoolMask, in_bbox)
        print('Creating Green Composite')
        GreenComp = CreateComposite(green_bands, NDVItmp, BoolMask, in_bbox)
        print('Creating Red Composite')
        RedComp = CreateComposite(red_bands, NDVItmp, BoolMask, in_bbox)
        print('Creating NIR Composite')
        NIRComp = CreateComposite(nir_bands, NDVItmp, BoolMask, in_bbox)
        print('Creating SWIR Composite')
        SWIRComp = CreateComposite(swir_bands, NDVItmp, BoolMask, in_bbox)
        print('Creating SWIR2 Composite')
        SWIR2Comp = CreateComposite(swir2_bands, NDVItmp, BoolMask, in_bbox)
        print('Creating NDVI Composite')
        NDVIComp = CollapseBands(NDVIstack, NDVItmp, BoolMask)
    
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
    stack = np.transpose([BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, NDVIComp, SAVI, MSAVI, NDMI, EVI, NBR, NBR2, TCB, TCG, TCW, ValidMask, Xgeo, Ygeo], [0, 1, 2]) 
    print("Assign band names")
    #assign band names
    bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'ValidMask', 'Xgeo', 'Ygeo']
    print("specifying output directory and filename")
    #outdir = '/projects/tmp/Landsat'
    outdir = args.output_dir
    out_file = os.path.join(outdir, 'Landsat8_' + str(tile_n) + '_comp_cog_2015-2020_dps.tif')
    
    # write COG to disk
    write_cog(stack, out_file, in_crs, crs_transform, bandnames, out_crs=out_crs, resolution=(res, res))

if __name__ == "__main__":
    '''
    Example call:
    python 3.1.2_dps.py -i /projects/maap-users/alexdevseed/boreal_tiles.gpkg -n 30543 -lyr boreal_tiles_albers  -o /projects/tmp/Landsat/ -b 0 -a https://landsatlook.usgs.gov/sat-api -l True
    '''
    main()
    
    

