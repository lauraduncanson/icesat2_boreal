# Search for imagery
# https://github.com/developmentseed/example-jupyter-notebooks/blob/landsat-search/notebooks/Landsat8-Search/L8-USGS-satapi.ipynb

import requests
import datetime
import geopandas as gpd
import folium
import shapely as shp

import json
import os
import boto3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import tarfile


from rasterio import enums
from rasterio.warp import array_bounds, calculate_default_transform
from rio_tiler.utils import create_cutline
from rio_cogeo.cogeo import cog_translate

import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import *
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio import windows
from rasterio.session import AWSSession
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rasterio.vrt import WarpedVRT
from rasterio.plot import show
from shapely.geometry import box
from fiona.crs import from_epsg

from CovariateUtils import get_index_tile

def write_local_data_and_catalog_s3(catalog, bands, save_path):
    '''Given path to a response json from a sat-api query, make a copy changing urls to local paths'''
    with open(catalog) as f:
        asset_catalog = json.load(f)
        for feature in asset_catalog['features']:
            #new_dir = os.path.join(save_path, feature['id'])
            #if (not os.path.isdir(new_dir)): os.mkdir(new_dir)
            #download the assests
            for band in bands:
                try:
                    key = feature['assets'][f'SR_{band}.TIF']['href'].replace(
                        'https://landsatlook.usgs.gov/data/',
                        '')
                    output_file = os.path.join(
                        f's3://maap-ops-dataset/alexdevseed/landsat8/sample2/{feature["id"]}/'
                        , os.path.basename(key))
                    #print(key)
                    ## Uncomment next line to actually download the data as a local sample
                    #s3.Bucket('usgs-landsat').download_file(key, output_file, ExtraArgs={'RequestPayer':'requester'})
                    feature['assets'][f'SR_{band}.TIF']['href'] = output_file
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print("The object does not exist.")
                    else:
                        raise
        # save and updated catalog with local paths
        local_catalog = catalog.replace('response', 'local-s3')
        with open(local_catalog,'w') as jsonfile:
            json.dump(asset_catalog, jsonfile)
        
        return local_catalog

def query_satapi(query, api):
    headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
            "Accept": "application/geo+json",
        }

    url = f"{api}/stac/search"
    data = requests.post(url, headers=headers, json=query).json()
    
    return data

def query_year(year, bbox, min_cloud, max_cloud, api):
    '''Given the year, finds the number of scenes matching the query and returns it.'''
    date_min = '-'.join([str(year), "06-01"])
    date_max = '-'.join([str(year), "09-15"])
    start_date = datetime.datetime.strptime(date_min, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(date_max, "%Y-%m-%d") 
    start = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end = end_date.strftime("%Y-%m-%dT23:59:59Z")
    
    query = {
    "time": f"{start}/{end}",
    "bbox":bbox,
    "query": {
        "collections": ["landsat-c2l2-sr"],
        "eo:platform": {"eq": "LANDSAT_8"},
        "eo:cloud_cover": {"gte": min_cloud, "lt": max_cloud},
        "landsat:collection_category":{"eq": "T1"}
        },
    "limit": 20 # We limit to 500 items per Page (requests) to make sure sat-api doesn't fail to return big features collection
    }
    
    data = query_satapi(query, api)
    
    return data



def get_data(in_tile_fn, in_tile_layer, in_tile_num, out_dir, sat_api):


    #geojson_path_albers = "/projects/maap-users/alexdevseed/boreal_tiles.gpkg"
    geojson_path_albers = in_tile_fn
    layer = in_tile_layer
    tile_n = int(in_tile_num)

    tile_id = get_index_tile(geojson_path_albers, tile_n, buffer=0, layer = layer)

    # Accessing imagery
    # Select an area of interest
    bbox_list = [tile_id['bbox_4326']]
    min_cloud = 0
    max_cloud = 20
    years = range(2015,2020 + 1)
    api = sat_api
    
    for bbox in bbox_list:
        # Geojson of total scenes - Change to list of scenes
        response_by_year = [query_year(year, bbox, min_cloud, max_cloud, api) for year in years]
        scene_totals = [each['meta']['found'] for each in response_by_year]
        print(scene_totals)
    
    # Take the search over several years, write the geojson response for each
    ## TODO: need unique catalog names that indicate bbox tile, and time range used.
    save_path = out_dir
    if (not os.path.isdir(save_path)): os.mkdir(save_path)
    catalogs = []
    for yr in range(0,len(years)):
        catalog = os.path.join(save_path, f'response-{tile_n}-{years[yr]}.json')
        with open(catalog, 'w') as jsonfile:
            json.dump(response_by_year[yr], jsonfile)
            catalogs.append(catalog)

    # create local versions
    bands = [''.join(["B",str(item)])for item in range(2,8,1)]     
    local_catalogs = [write_local_data_and_catalog_s3(catalog, bands, save_path) for catalog in catalogs]

    yr = 5
    scenes_poly = gpd.GeoDataFrame.from_features(response_by_year[yr], crs='epsg:4326')
    for col in scenes_poly.columns: print(col)
    
    return True
        

        
