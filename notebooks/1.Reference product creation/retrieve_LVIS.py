#!/usr/bin/env python
# coding: utf-8

# This script contains the function retrieveLVIS
# The function accepts a bounding box and returns LVIS granules from that location.

from maap.maap import MAAP
maap = MAAP(maap_host='api.ops.maap-project.org')

import ipycmc
w = ipycmc.MapCMC()
w

# import printing package to help display outputs
from pprint import pprint

# import rasterio for reading and writing in raster format
import rasterio as rio
# copy valid pixels from input files to an output file.
from rasterio.merge import merge
# set up AWS session
from rasterio.session import AWSSession
# display images, label axes
from rasterio.plot import show
# import boto3 to work with Amazon Web Services
import boto3

from io import StringIO
import pandas as pd

import os


# only run this block if needed in a new workspace
#!pip install -U folium geopandas rasterio>=1.2.3 rio-cogeo

def retrieveLVIS( granuleDomain ):
    #granuleDomain = "-147.51339,65.15264,-147.49457,65.15868"  # specify bounding box to search by


    # Earthdata search
    granule_results=maap.searchGranule(bounding_box=granuleDomain,concept_id="C1200235708-NASA_MAAP", limit=1000)
    pprint(f'Got {len(granule_results)} results')


    # speed up GDAL reads from S3 buckets by skipping sidecar files
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'


    # set up AWS session
    aws_session = AWSSession(boto3.Session())
    # get the S3 urls to the granules
    file_S3 = [item['Granule']['OnlineAccessURLs']['OnlineAccessURL']['URL'] for item in granule_results]
    # sort list in ascending order
    file_S3.sort()
    ## check list
    #file_S3

    #Split url string to get keys for LVIS granules
    itemlist = [i.split('s3://nasa-maap-data-store/', 1)[1] for i in file_S3]


    s3 = boto3.resource('s3')
    bucket = 'nasa-maap-data-store'

    lvis_result = []

    for key in itemlist:
        obj = s3.Object(bucket, key)
        body = obj.get()['Body'].read()
        s=str(body,'utf-8')
        data = StringIO(s) 
        df=pd.read_csv(data, error_bad_lines=False)
        lvis_result.append(df)
    
    return [lvis_result]



#lvis_result = pd.concat(lvis_result).reset_index(drop=True)






