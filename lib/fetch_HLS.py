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
maap = MAAP()

def write_local_data_and_catalog_s3(catalog, HLS_bands_dict, save_path, local, s3_path="s3://"):
    
    '''
    Given path to a response json from a sat-api query, make a copy changing urls to local paths
    updated: now works with a mix of HLS product types, because bands names (specific to each product type) 
    are retreived with a dictionary for each feature ID of the set of search results
    '''
    
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
        
        # Here is where you determine if you have S30 or L30 data, and specify the bands accordingly
        for feature in features:
            
            print(feature['id']) #  Check if feature can tell us if S30 or L30
            hls_product = feature['id'].split('.')[1]
            bands = HLS_bands_dict[hls_product]
            
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
        # Save and update catalog with local paths
        asset_catalog['features'] = clean_features
        local_catalog = catalog.replace('response', 'local-s3')
        with open(local_catalog,'w') as jsonfile:
            json.dump(asset_catalog, jsonfile)
        
        return local_catalog

def query_stac(year, bbox, max_cloud, api, start_month_day, end_month_day, MS_product='L30', MS_product_version='2.0', 
               MAX_N_RESULTS=100, MIN_N_FILT_RESULTS = 50, MAX_CLOUD_INC = 5, LIM_MAX_CLOUD = 90):
    
    print(f'\nQuerying STAC for multispectral imagery...')
    catalog = Client.open(api)
    print(f'Catalog title: {catalog.title}')
    
    date_min = str(year) + '-' + start_month_day

    date_max = str(year) + '-' + end_month_day
    start_date = datetime.datetime.strptime(date_min, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(date_max, "%Y-%m-%d") 
    start = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end = end_date.strftime("%Y-%m-%dT23:59:59Z")
    
    print('start date, end date:\t\t', start, end)
    
    # Note: H30 this is our name for a HARMONIZED 30m composite with S30 and L30
    # https://lpdaac.usgs.gov/news/important-update-to-cmr-stac-new-identifier-and-search-parameter-format/
    if MS_product == 'L30' or MS_product == 'S30':
        MS_product_list = [f"HLS{MS_product}_{MS_product_version}"]
    if MS_product == 'H30':
        MS_product_list = [f"HLSL30_{MS_product_version}", f"HLSS30_{MS_product_version}"]
    if MS_product == 'landsat-c2l2-sr':
        MS_product_list = [MS_product]
        
    print(f"\nConducting multispectral image search now ...")
    print(f"Searching for:\t\t\t{MS_product_list}")
    print(f"Max cloudcover threshold starts at: {max_cloud}% and won't exceed {LIM_MAX_CLOUD}%")
    print(f"Min number of filtered results: {MIN_N_FILT_RESULTS}")

    while True:
        results_list = []
        # Doing this loop to get around CMR bug: https://github.com/nasa/cmr-stac/pull/357
        # Loop can be removed and product list inserted back into 'collections' parameter when fixed
        for MS_prod in MS_product_list:
            
            search = catalog.search(
                    collections=MS_prod,
                    datetime=[start , end],
                    bbox = bbox,
                    limit=MAX_N_RESULTS,
                    max_items=None
                    ,query={"eo:cloud_cover":{"lte":max_cloud}} # used to not work..now it does
                )
            results = search.get_all_items_as_dict()
            print(f"partial results ({MS_prod}):\t\t\t\t{len(results['features'])}")
            results_list.append(results)

        # This flattens the results list as well as doing a secondary cloud_cover filtering
        filtered_results = []
        for results in results_list:
            for i in results['features']:
                if int(i['properties']['eo:cloud_cover']) <= max_cloud: # this filter can be removed now
                    filtered_results.append(i)
                    
        # Get the # of filtered results: the total # of HLS scenes returned from search           
        N_FILT_RESULTS = len(filtered_results)
        if N_FILT_RESULTS >= MIN_N_FILT_RESULTS or max_cloud > LIM_MAX_CLOUD - MAX_CLOUD_INC:
            break
        max_cloud += MAX_CLOUD_INC
        print(f"\tOnly {N_FILT_RESULTS} HLS scenes using lte {max_cloud} cloudcover for {start}-{end} for bbox.")
        print(f"\tIncrease max_cloud by {MAX_CLOUD_INC}% until you get >= {MIN_N_FILT_RESULTS} scenes to composite or the {LIM_MAX_CLOUD}% cloudcover limit is reached.")
        print(f"Max cloudcover threshold now at: {max_cloud}%")

    results['features'] = filtered_results

    print(f"compelte results ({MS_product_list}):\t{len(results['features'])}")
    print('\nSearch complete.\n')
    return results

def get_HLS_data(in_tile_fn, in_tile_layer, in_tile_id_col, in_tile_num, out_dir, sat_api, 
                 start_year, end_year, start_month_day, end_month_day, max_cloud, local=False, 
                 hls_product='L30', hls_product_version='2.0', min_n_filt_results=0):

    # Need a dict that used HLS product to specify band names
    HLS_bands_dict = dict({
                            'L30': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask'], 
                            'S30': ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12', 'Fmask']
                          })
    
    if hls_product != 'L30' and hls_product != 'S30' and hls_product != 'H30':
        print("HLS product type not recognized: Must be L30, S30, or both [H30].")
        os._exit(1)
        
    geojson_path_albers = in_tile_fn
    layer = in_tile_layer
    tile_n = int(in_tile_num)
    
    print('\nGetting HLS data...')
    
    tile_id = get_index_tile(geojson_path_albers, in_tile_id_col, tile_n, buffer=0, layer = layer)
    #print(tile_id)
    # Accessing imagery
    # Select an area of interest
    bbox_list = [tile_id['bbox_4326']]
    max_cloud = max_cloud
    years = range(int(start_year), int(end_year)+1)
    api = sat_api
    
    #
    # Query the STAC
    #
    for bbox in bbox_list:
        # Geojson of total scenes - Change to list of scenes
        print(f'bbox: {bbox}')
        response_by_year = [query_stac(year, bbox, max_cloud, api, start_month_day, end_month_day, MS_product=hls_product, 
                                       MS_product_version=hls_product_version, MIN_N_FILT_RESULTS=min_n_filt_results) for year in years]
        
        print(len(response_by_year[0]['features']))
    
    # Take the search over several years, write the geojson response for each
    ## TODO: need unique catalog names that indicate bbox tile, and time range used.
    save_path = out_dir
    if (not os.path.isdir(save_path)): os.mkdir(save_path)

    merge_catalogs = {
        "type": "FeatureCollection",
        "features": list(itertools.chain.from_iterable([f["features"] for f in response_by_year])),
    }

    #
    # Write local JSON that catalogs the HLS data retrieved from query
    #
    master_json = os.path.join(save_path, f'master_{tile_n}_{np.min(years)}-{start_month_day}_{np.max(years)}-{end_month_day}_HLS.json')
    with open(master_json, 'w') as outfile:
            json.dump(merge_catalogs, outfile)

    master_json = write_local_data_and_catalog_s3(master_json, HLS_bands_dict, save_path, local, s3_path="s3://")
    
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