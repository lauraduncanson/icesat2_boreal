# Search for imagery
# https://github.com/developmentseed/example-jupyter-notebooks/blob/landsat-search/notebooks/Landsat8-Search/L8-USGS-satapi.ipynb

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


def write_local_data_and_catalog_s3(catalog, bands, save_path, local, s3_path="s3://maap-ops-dataset/maap-users/alexdevseed/landsat8/sample2/"):
    '''Given path to a response json from a sat-api query, make a copy changing urls to local paths'''
    creds = get_creds()
    aws_session = boto3.session.Session(
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'])
    s3 = aws_session.client('s3')
    
    with open(catalog) as f:
        clean_features = []
        asset_catalog = json.load(f)
        
        # Remove duplicate scenes, keeping newest
        features = asset_catalog['features']
        sorted_features = sorted(features, key=lambda f: (f["properties"]["landsat:scene_id"], f["id"]))
        most_recent_features = list({ f["properties"]["landsat:scene_id"]: f for f in sorted_features }.values())
        
        for feature in most_recent_features:
            #print(feature)
            try:
                for band in bands:
                    if local:
                        key = feature['assets'][band]['href'].replace('https://landsatlook.usgs.gov/data/','')
                        output_file = os.path.join(f'{s3_path}{feature["id"][:-3]}/', os.path.basename(key))
                        
                    else:
                        output_file = feature['assets'][band]['href'].replace('https://landsatlook.usgs.gov/data/', s3_path)
                    # Only update the url to s3 if the s3 file exists
                    #print(output_file)
                    bucket_name = output_file.split("/")[2]
                    s3_key = "/".join(output_file.split("/")[3:])
                    head = s3.head_object(Bucket = bucket_name, Key = s3_key, RequestPayer='requester')
                    if head['ResponseMetadata']['HTTPStatusCode'] == 200:
                        feature['assets'][band]['href'] = output_file
                clean_features.append(feature)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print(f"The object does not exist. {output_file}")
                else:
                    raise
        # save and updated catalog with local paths
        asset_catalog['features'] = clean_features
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

def query_year(year, bbox, max_cloud, api, start_month_day, end_month_day):
    '''Given the year, finds the number of scenes matching the query and returns it.'''
    date_min = str(year) + '-' + start_month_day
    date_max = str(year) + '-' + end_month_day
    start_date = datetime.datetime.strptime(date_min, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(date_max, "%Y-%m-%d") 
    start = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end = end_date.strftime("%Y-%m-%dT23:59:59Z")
    
    print('start date, end date = ', start, end)
    
    print('conducting search now')
    
    query = {
    "time": f"{start}/{end}",
    "bbox":bbox,
    "query": {
        "collections": ["landsat-c2l2-sr"],
        "platform": {"in": ["LANDSAT_8"]},
        "eo:cloud_cover": {"gte": 0, "lt": max_cloud},
        "landsat:collection_category":{"in": ["T1"]}
        },
    "limit": 20 # We limit to 500 items per Page (requests) to make sure sat-api doesn't fail to return big features collection
    }
    
    data = query_satapi(query, api)
    
    return data

def read_json(json_file):
    with open(json_file) as f:
        response = json.load(f)
    return response

def get_ls8_data(in_tile_fn, in_tile_layer, in_tile_id_col, in_tile_num, out_dir, sat_api, start_year, end_year, start_month_day, end_month_day, max_cloud, local=False):

    geojson_path_albers = in_tile_fn
    layer = in_tile_layer
    tile_n = int(in_tile_num)

    tile_id = get_index_tile(geojson_path_albers, in_tile_id_col, tile_n, buffer=0, layer = layer)
    print(tile_id)
    # Accessing imagery
    # Select an area of interest
    bbox_list = [tile_id['bbox_4326']]
    max_cloud = max_cloud
    years = range(int(start_year), int(end_year)+1)
    print(years)
    api = sat_api
    
    for bbox in bbox_list:
        # Geojson of total scenes - Change to list of scenes
        response_by_year = [query_year(year, bbox, max_cloud, api, start_month_day, end_month_day) for year in years]
        scene_totals = [each['meta']['found'] for each in response_by_year]
        print('scene total: ', scene_totals)
    
    # Take the search over several years, write the geojson response for each
    ## TODO: need unique catalog names that indicate bbox tile, and time range used.
    save_path = out_dir
    if (not os.path.isdir(save_path)): os.mkdir(save_path)

    merge_catalogs = {
        "type": "FeatureCollection",
        "features": list(itertools.chain.from_iterable([f["features"] for f in response_by_year])),
    }
        
    master_json = os.path.join(save_path, f'master_{tile_n}_{np.min(years)}-{start_month_day}_{np.max(years)}-{end_month_day}_LS8.json')
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
        

        
