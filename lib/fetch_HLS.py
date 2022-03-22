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
from pystac_client import Client
from maap.maap import MAAP
maap = MAAP(maap_host='api.ops.maap-project.org')


def write_local_data_and_catalog_s3(catalog, bands, save_path, local, s3_path="s3://"):
    '''Given path to a response json from a sat-api query, make a copy changing urls to local paths'''
    creds = maap.aws.earthdata_s3_credentials('https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials')
    aws_session = boto3.session.Session(
        aws_access_key_id=creds['accessKeyId'], 
        aws_secret_access_key=creds['secretAccessKey'],
        aws_session_token=creds['sessionToken'],
        region_name='us-west-2')
    s3 = aws_session.client('s3')
    
    with open(catalog) as f:
        clean_features = []
        asset_catalog = json.load(f)
        
        
        
        # Remove duplicate scenes, keeping newest
        features = asset_catalog['features']
        #sorted_features = sorted(features, key=lambda f: (f["properties"]["landsat:scene_id"], f["id"]))
        #most_recent_features = list({ f["properties"]["landsat:scene_id"]: f for f in sorted_features }.values())
        
        for feature in features:
            #print(feature)
            try:
                for band in bands:
                    output_file = feature['assets'][band]['href'].replace('https://data.lpdaac.earthdatacloud.nasa.gov/', s3_path)
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

def query_stac(year, bbox, max_cloud, api, start_month_day, end_month_day):
    print('opening client')
    catalog = Client.open(api)
    
    date_min = str(year) + '-' + start_month_day
    print('start_month_day:\t\t', start_month_day)
    date_max = str(year) + '-' + end_month_day
    start_date = datetime.datetime.strptime(date_min, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(date_max, "%Y-%m-%d") 
    start = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end = end_date.strftime("%Y-%m-%dT23:59:59Z")
    
    print('start date, end date:\t\t', start, end)
    
    print('\nConducting HLS search now...')
    
    search = catalog.search(
        collections=["HLSL30.v2.0"],
        datetime=[start,end],
        bbox=bbox,
        max_items=500, # for testing, and keep it from hanging
        # query={"eo:cloud_cover":{"lt":20}} #doesn't work
    )
    print(f"Search query parameters:\n{search}\n")
    results = search.get_all_items_as_dict()
    
    print("initial results:\t\t", len(results['features']))
    
    filtered_results = []
    for i in results['features']:
        if int(i['properties']['eo:cloud_cover']) <= max_cloud:
            filtered_results.append(i)
    
    results['features'] = filtered_results

    print("filtered results:\t\t", len(results['features']))
    print('\nSearch complete.\n')
    return results


def get_HLS_data(in_tile_fn, in_tile_layer, in_tile_id_col, in_tile_num, out_dir, sat_api, start_year, end_year, start_month_day, end_month_day, max_cloud, local=False):

    geojson_path_albers = in_tile_fn
    layer = in_tile_layer
    tile_n = int(in_tile_num)

    tile_id = get_index_tile(geojson_path_albers, in_tile_id_col, tile_n, buffer=0, layer = layer)
    #print(tile_id)
    # Accessing imagery
    # Select an area of interest
    bbox_list = [tile_id['bbox_4326']]
    max_cloud = max_cloud
    years = range(int(start_year), int(end_year)+1)
    api = sat_api
    
    for bbox in bbox_list:
        # Geojson of total scenes - Change to list of scenes
        print('run function')
        response_by_year = [query_stac(year, bbox, max_cloud, api, start_month_day, end_month_day) for year in years]
        
        print(len(response_by_year[0]['features']))
    
    # Take the search over several years, write the geojson response for each
    ## TODO: need unique catalog names that indicate bbox tile, and time range used.
    save_path = out_dir
    if (not os.path.isdir(save_path)): os.mkdir(save_path)

    merge_catalogs = {
        "type": "FeatureCollection",
        "features": list(itertools.chain.from_iterable([f["features"] for f in response_by_year])),
    }
        
    master_json = os.path.join(save_path, f'master_{tile_n}_{np.min(years)}-{start_month_day}_{np.max(years)}-{end_month_day}_HLS.json')
    with open(master_json, 'w') as outfile:
            json.dump(merge_catalogs, outfile)
    
    bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask']
    # If local True, rewrite the s3 paths to internal not public buckets
    #bands = [''.join(["B",str(item)])for item in range(2,8,1)]
    master_json = write_local_data_and_catalog_s3(master_json, bands, save_path, local, s3_path="s3://")
    
    return master_json

    
'''
if __name__ == "__main__":
    in_tile_fn = '/projects/shared-buckets/nathanmthomas/boreal_grid_albers90k_gpkg.gpkg'
    in_tile_layer = 'grid_boreal_albers90k_gpkg'
    in_tile_num = 3013
    out_dir = '/projects/Developer/icesat2_boreal/notebooks/3.Gridded_product_development/'
    sat_api = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"

    data = get_data(in_tile_fn, in_tile_layer, in_tile_num, out_dir, sat_api, local=False)
'''