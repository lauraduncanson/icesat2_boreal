import requests
import datetime
import geopandas as gpd
import json
import os
import numpy as np
import rasterio as rio
from rasterio.warp import *
from CovariateUtils import get_index_tile, get_creds, get_creds_DAAC
import itertools
import botocore
import boto3
from pystac_client import Client
from maap.maap import MAAP
maap = MAAP()

# Set a dict with keys of all products to specify their band names
MS_BANDS_DICT = dict({
                        'L30':  ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask'], 
                        'S30':  ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12', 'Fmask'],
                        'LT04': ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22','cloud_qa'],
                        'LT05': ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22','cloud_qa'],
                        'LE07': ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22','cloud_qa'],
                        'LC08': ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22','qa_pixel'],
                        'LC09': ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22','qa_aerosol']
                      })

def write_local_data_and_catalog_s3(catalog, bands_dict, comp_type, save_path, local, s3_path="s3://maap-ops-dataset/maap-users/montesano/sample/"):
    
    '''Given path to a response json from a STAC api query, make a copy changing urls to local or s3 paths'''
    
    if comp_type == 'HLS':
        PROD_SPLITTER, SPLIT_IDX = ('.', 1)
        creds = get_creds_DAAC()
        aws_session = boto3.session.Session(
                aws_access_key_id=creds['accessKeyId'],
                aws_secret_access_key=creds['secretAccessKey'],
                aws_session_token=creds['sessionToken']
    )
    elif comp_type == 'LC2SR':
        PROD_SPLITTER, SPLIT_IDX = ('_', 0)
        creds = get_creds()
        aws_session = boto3.session.Session(
                aws_access_key_id=creds['AccessKeyId'],
                aws_secret_access_key=creds['SecretAccessKey'],
                aws_session_token=creds['SessionToken']
    )
    else:
        print(f"Attempting to split file name but composite type ({comp_type}) not recognized.")
        os._exit(1)
    
    s3 = aws_session.client('s3')
    
    with open(catalog) as f:
        clean_features = []
        asset_catalog = json.load(f)
        
        features = asset_catalog['features']
        ## Remove duplicate scenes, keeping newest
        # sorted_features = sorted(features, key=lambda f: (f["properties"]["landsat:scene_id"], f["id"]))
        # most_recent_features = list({ f["properties"]["landsat:scene_id"]: f for f in sorted_features }.values())
            
        for feature in features:
            print(feature['id']) #Print the name of each granule returned from search that you'll use for composite
            product = feature['id'].split(PROD_SPLITTER)[SPLIT_IDX]
            bands = bands_dict[product]
            try:
                for band in bands:
                    if comp_type == 'LC2SR':
                        if local:
                            key = feature['assets'][band]['href'].replace('https://landsatlook.usgs.gov/data/','')
                            output_file = os.path.join(f'{s3_path}{feature["id"][:-3]}/', os.path.basename(key))
                        else:
                            output_file = feature['assets'][band]['href'].replace('https://landsatlook.usgs.gov/data/', s3_path)
                    elif comp_type == 'HLS':
                        output_file = feature['assets'][band]['href'].replace('https://data.lpdaac.earthdatacloud.nasa.gov/', s3_path)
                    else:
                        print(f"Composite type ({comp_type}) not recognized")
                        os._exit(1)
                    
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
                    
        # save and update catalog with local or s3 paths
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
    
    # Note: H30 this is our name for a HARMONIZED 30m composite. It indicates that we want both S30 and L30
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
        if True:
            '''Use pystac_client to query the API for HLS file paths
            '''
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
                
        if False:
            '''Use rustac to query local copy (*.parquet) of HLS file paths
                Here need to get search of the *.parquet to return a results_list like what is returned after catalog.search() above
            '''
            from rustac import DuckdbClient
            client = DuckdbClient(use_hive_partitioning=True)
            
            # configure duckdb to find S3 credentials
            client.execute(
                """
                CREATE OR REPLACE SECRET secret (
                     TYPE S3,
                     PROVIDER CREDENTIAL_CHAIN
                );
                """
            )
            
            # use rustac/duckdb to search through the partitioned parquet dataset to find matching items
            search = client.search(
                #collections='HLSL30_2.0',
                href="s3://maap-ops-workspace/shared/henrydevseed/hls-stac-geoparquet-v1/year=*/month=*/*.parquet",
                datetime="2024-07-01T00:00:00Z/2024-08-31T23:59:59Z",
                bbox=bbox,
            )
            ###results = search.get_all_items_as_dict()
            results = dict()
            results['features'] = search
            #print(f"partial results ({MS_prod}):\t\t\t\t{len(results['features'])}")
            print(f"partial results (---all ---):\t\t\t\t{len(results['features'])}")
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
        print(f"\tOnly {N_FILT_RESULTS} HLS scenes using lte {max_cloud}% cloudcover for {start}-{end} for bbox.")
        print(f"\tIncrease max_cloud by {MAX_CLOUD_INC}% until you get >= {MIN_N_FILT_RESULTS} scenes to composite or the {LIM_MAX_CLOUD}% cloudcover limit is reached.")
        print(f"Max cloudcover threshold now at: {max_cloud}%")

    results['features'] = filtered_results

    print(f"compelte results ({MS_product_list}):\t{len(results['features'])}")
    print('\nSearch complete.\n')
    return results
        
def get_ms_data(in_tile_fn, in_tile_layer, in_tile_id_col, in_tile_num, out_dir, sat_api, 
                 start_year, end_year, start_month_day, end_month_day, max_cloud, 
                       comp_type,
                       local=False, 
                    ms_product='H30', hls_product_version='2.0', 
                    min_n_filt_results=0, max_cloud_inc=5, bands_dict=MS_BANDS_DICT):
      
    geojson_path_albers = in_tile_fn
    layer = in_tile_layer
    tile_n = int(in_tile_num)
    
    if comp_type == 'LC2SR':
        print(f'\nGetting Landsat Collection 2 Surface Reflectance ({ms_product}) data...')
        s3_path = "s3://usgs-landsat/"
    if comp_type == 'HLS':
        print(f'\nGetting HLS Surface Reflectance {ms_product} data...')
        s3_path = "s3://"
    
    tile_id = get_index_tile(geojson_path_albers, in_tile_id_col, tile_n, buffer=0, layer = layer)
    bbox_list = [tile_id['bbox_4326']]
    max_cloud = max_cloud
    years = range(int(start_year), int(end_year)+1)
    api = sat_api
    
    ################
    # Query the STAC
    #
    for bbox in bbox_list:
        # Geojson of total scenes - Change to list of scenes
        print(f'bbox: {bbox}')
        response_by_year = [query_stac(year, bbox, max_cloud, api, start_month_day, end_month_day, MS_product=ms_product, 
                                       MS_product_version=hls_product_version, MIN_N_FILT_RESULTS=min_n_filt_results, MAX_CLOUD_INC=max_cloud_inc) for year in years]
        
        print(len(response_by_year[0]['features']))
    
    # Take the search over several years, write the geojson response for each
    save_path = out_dir
    if (not os.path.isdir(save_path)): os.mkdir(save_path)

    merge_catalogs = {
        "type": "FeatureCollection",
        "features": list(itertools.chain.from_iterable([f["features"] for f in response_by_year])),
    }

    ################
    # Write local JSON that catalogs the data retrieved from STAC query
    #
    master_json = os.path.join(save_path, f'master_{tile_n}_{np.min(years)}-{start_month_day}_{np.max(years)}-{end_month_day}_{comp_type}.json')
    with open(master_json, 'w') as outfile:
            json.dump(merge_catalogs, outfile)

    master_json = write_local_data_and_catalog_s3(master_json, bands_dict, comp_type, save_path, local, s3_path=s3_path)
    
    return master_json
        
# # Old functions
# def query_satapi(query, api):
#     headers = {
#             "Content-Type": "application/json",
#             "Accept-Encoding": "gzip",
#             "Accept": "application/geo+json",
#         }

#     url = f"{api}/stac/search"
#     print(f'\nQuerying api: {url}\n')
#     data = requests.post(url, headers=headers, json=query).json()
    
#     return data

# def query_year(year, bbox, max_cloud, api, start_month_day, end_month_day):
#     '''Given the year, finds the number of scenes matching the query and returns it.'''
#     date_min = str(year) + '-' + start_month_day
#     date_max = str(year) + '-' + end_month_day
#     start_date = datetime.datetime.strptime(date_min, "%Y-%m-%d")
#     end_date = datetime.datetime.strptime(date_max, "%Y-%m-%d") 
#     start = start_date.strftime("%Y-%m-%dT00:00:00Z")
#     end = end_date.strftime("%Y-%m-%dT23:59:59Z")
    
#     print('start date, end date:\t\t', start, end)
    
#     print('\nConducting Landsat 8 search now...')
    
#     query = {
#     "time": f"{start}/{end}",
#     "bbox":bbox,
#     "query": {
#         "collections": ["landsat-c2l2-sr"],
#         "platform": {"in": ["LANDSAT_8"]},
#         "eo:cloud_cover": {"gte": 0, "lt": max_cloud},
#         "landsat:collection_category":{"in": ["T1"]}
#         },
#     "limit": 500 # We limit to 500 items per Page (requests) to make sure sat-api doesn't fail to return big features collection
#     }
    
#     print(f"Search query parameters:\n{query}\n")
    
#     data = query_satapi(query, api)
#     print('\nSearch complete.\n')
#     return data

# def read_json(json_file):
#     with open(json_file) as f:
#         response = json.load(f)
#     return response

# def get_ls8_data(in_tile_fn, in_tile_layer, in_tile_id_col, in_tile_num, out_dir, sat_api, start_year, end_year, start_month_day, end_month_day, max_cloud, local=False):

#     geojson_path_albers = in_tile_fn
#     layer = in_tile_layer
#     tile_n = int(in_tile_num)

#     tile_id = get_index_tile(geojson_path_albers, in_tile_id_col, tile_n, buffer=0, layer = layer)
#     #print(f"\nPrinting tile parts:\n\t{tile_id}")
#     # Accessing imagery
#     # Select an area of interest
#     bbox_list = [tile_id['bbox_4326']]
#     max_cloud = max_cloud
#     years = range(int(start_year), int(end_year)+1)
#     print(years)
#     api = sat_api
    
#     for bbox in bbox_list:
#         # Geojson of total scenes - Change to list of scenes
#         response_by_year = [query_year(year, bbox, max_cloud, api, start_month_day, end_month_day) for year in years]
#         scene_totals = [each['meta']['found'] for each in response_by_year]
#         print('scene total: ', scene_totals)
    
#     # Take the search over several years, write the geojson response for each
#     ## TODO: need unique catalog names that indicate bbox tile, and time range used.
#     save_path = out_dir
#     if (not os.path.isdir(save_path)): os.mkdir(save_path)

#     merge_catalogs = {
#         "type": "FeatureCollection",
#         "features": list(itertools.chain.from_iterable([f["features"] for f in response_by_year])),
#     }
        
#     master_json = os.path.join(save_path, f'master_{tile_n}_{np.min(years)}-{start_month_day}_{np.max(years)}-{end_month_day}_LS8.json')
#     with open(master_json, 'w') as outfile:
#             json.dump(merge_catalogs, outfile)
    
#     bands = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']
#     # If local True, rewrite the s3 paths to internal not public buckets
#     if (local):
#         # create local versions, only for the bands we use currently
#         #bands = [''.join(["B",str(item)])for item in range(2,8,1)]
#         master_json = write_local_data_and_catalog_s3(master_json, bands, save_path, local, s3_path="s3://maap-ops-dataset/maap-users/alexdevseed/landsat8/sample2/")
#     else:
#         #bands = [''.join(["B",str(item)])for item in range(2,8,1)]
#         master_json = write_local_data_and_catalog_s3(master_json, bands, save_path, local, s3_path="s3://usgs-landsat/")
#     return master_json