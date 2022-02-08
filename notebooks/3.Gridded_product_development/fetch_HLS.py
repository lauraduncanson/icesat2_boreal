import requests
import datetime
import geopandas as gpd
import json
import os
import numpy as np
import rasterio as rio
from rasterio.warp import *
from CovariateUtils import get_index_tile, get_creds
import itertools
import botocore
import boto3





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
        "collections": ["HLSL30.v2.0"],
        #"platform": {"in": ["LANDSAT_8"]},
        "eo:cloud_cover": {"gte": min_cloud, "lt": max_cloud},
        #"landsat:collection_category":{"in": ["T1"]}
        },
    "limit": 20 # We limit to 500 items per Page (requests) to make sure sat-api doesn't fail to return big features collection
    }
    
    data = query_satapi(query, api)
    
    return data

def get_data(in_tile_fn, in_tile_layer, in_tile_num, out_dir, sat_api, local=False):

    geojson_path_albers = in_tile_fn
    layer = in_tile_layer
    tile_n = int(in_tile_num)

    tile_id = get_index_tile(geojson_path_albers, tile_n, buffer=0, layer = layer)
    print(tile_id)
    # Accessing imagery
    # Select an area of interest
    bbox_list = [tile_id['bbox_4326']]
    min_cloud = 0
    max_cloud = 20
    # 2015
    years = range(2015,2020 + 1)
    api = sat_api
    
    for bbox in bbox_list:
        # Geojson of total scenes - Change to list of scenes
        response_by_year = [query_year(year, bbox, min_cloud, max_cloud, api) for year in years]
        print(response_by_year)
        scene_totals = [each['meta']['found'] for each in response_by_year]
        print('scene total: ', scene_totals)
    '''
    # Take the search over several years, write the geojson response for each
    ## TODO: need unique catalog names that indicate bbox tile, and time range used.
    save_path = out_dir
    if (not os.path.isdir(save_path)): os.mkdir(save_path)

    merge_catalogs = {
        "type": "FeatureCollection",
        "features": list(itertools.chain.from_iterable([f["features"] for f in response_by_year])),
    }
        
    master_json = os.path.join(save_path, f'master-{tile_n}-{np.min(years)}-{np.max(years)}.json')
    with open(master_json, 'w') as outfile:
            json.dump(merge_catalogs, outfile)
    
    bands = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']
    # If local True, rewrite the s3 paths to internal not public buckets
    if (local):
        # create local versions, only for the bands we use currently
        #bands = [''.join(["B",str(item)])for item in range(2,8,1)]
        master_json = write_local_data_and_catalog_s3(master_json, bands, save_path, local, s3_path="s3://maap-ops-dataset/maap-users/alexdevseed/landsat8/sample2/")
    else:
        #bands = [''.join(["B",str(item)])for item in range(2,8,1)]
        master_json = write_local_data_and_catalog_s3(master_json, bands, save_path, local, s3_path="s3://usgs-landsat/")
    return master_json

    '''


in_tile_fn = '/projects/shared-buckets/nathanmthomas/boreal_grid_albers90k_gpkg.gpkg'
in_tile_layer = 'grid_boreal_albers90k_gpkg'
in_tile_num = 3013
out_dir = '/projects/tmp/Landsat/TC_test'
sat_api = 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search'

data = get_data(in_tile_fn, in_tile_layer, in_tile_num, out_dir, sat_api, local=False)