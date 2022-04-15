import pandas as pd
import geopandas
import rasterio as rio
from rasterio.crs import CRS
import os
import numpy as np

import sys
#sys.path.append('/projects/code/icesat2_boreal/notebooks/3.Gridded_product_development')
#from CovariateUtils import *
#import CovariateUtils

import itertools
import copy

#from notebooks.general.covariateutils import get_index_tile
#import notebooks.general.CovariateUtils 
#from CovariateUtils import get_index_tile

try:
    from maap.maap import MAAP
    # create MAAP class
    maap = MAAP(maap_host='api.ops.maap-project.org')
    HAS_MAAP = True
except ImportError:
    print('NASA MAAP is unavailable')
    HAS_MAAP = False

def get_index_tile(vector_path: str, id_col: str, tile_id: int, buffer: float = 0, layer: str = None):
    '''
    Given a vector tile index, select by id the polygon and return
    GPKG is the recommended vector format - single file, includes projection, can contain multiple variants and additional information.
    TODO: should it be a class or dict
    
    
    vector_path: str
        Path to GPKG file
    buffer: float
        Distance to buffer geometry in units of layer
    id_col: str
        Column name of the tile_id
    tile_id: int
        Tile ID to extract/build info for
        
    returns:
        geopandas.geodataframe.GeoDataFrame,
            Polygon in original crs
        geopandas.geoseries.GeoSeries,
            Polygon of Buffered in original crs
        list,
            Bounds of original polygon
        rasterio.crs.CRS,
            Coordinate Reference System of original tile
        geopandas.geodataframe.GeoDataFrame,
            4326 Polygon
        list,
            Bounds in 4326
        geopandas.geoseries.GeoSeries,
            Polygon of Buffered in 4326
        list
            Buffered Bounds in 4326
    Usage:
    get_index_tile(
        vector_path = '/projects/maap-users/alexdevseed/boreal_tiles.gpkg',
        tile_id = 30542,
        buffer = 120
        )
    
    '''
    
    tile_parts = {}

    if layer is None:
        layer = os.path.splitext(os.path.basename(vector_path))[0]
    tile_index = geopandas.read_file(vector_path, layer=layer)

    tile_parts["geom_orig"] = tile_index[tile_index[id_col]==tile_id]
    tile_parts["geom_orig_buffered"] = tile_parts["geom_orig"]["geometry"].buffer(buffer)
    tile_parts["bbox_orig"] = tile_parts["geom_orig"].bounds.iloc[0].to_list()
    tile_parts["tile_crs"] = CRS.from_wkt(tile_index.crs.to_wkt()) #A rasterio CRS object

    # Properties of 4326 version of tile
    tile_parts["geom_4326"] = tile_parts["geom_orig"].to_crs(4326)
    tile_parts["bbox_4326"] = tile_parts["geom_4326"].bounds.iloc[0].to_list()
    tile_parts["geom_4326_buffered"] =  tile_parts["geom_orig_buffered"].to_crs(4326)
    tile_parts["bbox_4326_buffered"] = tile_parts["geom_4326_buffered"].bounds.iloc[0].to_list()
    
    return tile_parts

def maap_search_get_h5_list(tile_num, tile_fn="/projects/shared-buckets/nathanmthomas/boreal_tiles_v003.gpkg", layer="boreal_tiles_v003", id_col="tile_num", DATE_START='06-01', DATE_END='09-30', YEARS=[2019, 2020, 2021], version=4, MAX_GRANULES=10000):
    '''
    Return a list of ATL08 h5 names that intersect a tile for a give date range across a set of years
    '''
    tile_parts = get_index_tile(tile_fn, id_col, tile_num, buffer=0, layer=layer)

    in_bbox = ",".join(str(coord) for coord in tile_parts['bbox_4326'])
    
    print("\tTILE_NUM: {} ({})".format(tile_num, in_bbox) )
    
    out_crs = tile_parts['tile_crs']
    
    DATE_START = DATE_START + 'T00:00:00Z' # SUMMER start
    DATE_END = DATE_END + 'T23:59:59Z' # SUMMER end
    
    date_filters = [f'{year}-{DATE_START},{year}-{DATE_END}' for year in YEARS]
    version = str(f'{version:03}')

    base_query = {
    'short_name':"ATL08",
    'version':version, 
    'bounding_box':in_bbox,
    'limit': MAX_GRANULES,
    }

    #q3 = [build_query(copy.copy(base_query), date_filter) for date_filter in date_filters]
    queries = [dict(base_query, temporal=date_filter) for date_filter in date_filters]
    print(f"\tSearching MAAP for granules using these parameters: \n\t{queries}")
    
    # query CMR as many seasons as necessary
    result_chain = itertools.chain.from_iterable([maap.searchGranule(**query) for query in queries])
    
    # This is the list of ATL08 that intersect the tile bounds
    # Use this list of ATL08 to identify the ATL08 h5/CSV files you have already DPS'd.
    # get the s3 urls for granules we want to process
    granules = [item.getDownloadUrl() for item in result_chain]
    
    # Convert to just the h5 basenames (removing the s3 url)
    out_h5_list = [os.path.basename(x) for x in granules]
    
    print("\t\t# ATL08 for tile {}: {}".format(tile_num, len(out_h5_list)) )
    
    return(out_h5_list)

def extract_value_gdf(r_fn, pt_gdf, bandnames: list, reproject=True, TEST=False):
    """Extract raster band values to the obs of a geodataframe
    """

    print("\tExtracting raster values from: ", r_fn)
    r_src = rio.open(r_fn)
    
    if reproject:
        print("\tRe-project points to match raster...")
        pt_gdf = pt_gdf.to_crs(r_src.crs)
    
    for idx, bandname in enumerate(bandnames):
        bandnum = idx + 1
        if TEST: print("Read as a numpy masked array...")
        r = r_src.read(bandnum, masked=True)
        
        if TEST: print(r.dtype)

        pt_coord = [(pt.x, pt.y) for pt in pt_gdf.geometry]

        # Use 'sample' from rasterio
        if TEST: print("Create a generator for sampling raster...")
        pt_sample = r_src.sample(pt_coord, bandnum)
        
        if TEST:
            for i, val in enumerate(r_src.sample(pt_coord, bandnum)):
                print("point {} value: {}".format(i, val))
            
        if TEST: print("Use generator to evaluate (sample)...")
        pt_sample_eval = np.fromiter(pt_sample, dtype=r.dtype)

        if TEST: print("Deal with no data...")
        pt_sample_eval_ma = np.ma.masked_equal(pt_sample_eval, r_src.nodata)
        #pt_gdf[bandname] = pd.Categorical(pt_sample_eval_ma.astype(int).filled(-1))
        pt_gdf[bandname] = pt_sample_eval_ma.astype(float).filled(np.nan)
        
        #print('\tDataframe has new raster value column: {}'.format(bandname))
        r = None
        
    r_src.close()
    
    print('\tReturning {} points with {} new raster value columns: {}'.format(len(pt_gdf), len(bandnames), bandnames))
    return(pt_gdf)

def get_covar_fn_list(rootDir, tile_num):
    '''
    Get a list of covar filenames using a root dir and a tile_num string that is found in each covar file name
    '''
    covar_tile_list = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if fname.endswith('.tif') and str(tile_num) in fname:
                covar_tile_list.append(os.path.join(dirName , fname))
                
    return(covar_tile_list)

def add_neighbors_gdf(input_gdf, input_id_field):
    input_gdf["NEIGHBORS"] = None
    for index, feature in input_gdf.iterrows():   
        #print(tile.tile_num)
        # get 'not disjoint' countries
        neighbors = input_gdf[~input_gdf.geometry.disjoint(feature.geometry)][input_id_field].tolist()
        #print(feature.tile_num)
        # remove own name of the country from the list
        neighbors = [ fid for fid in neighbors if feature[input_id_field] != fid ]
        #print(neighbors)

        # https://stackoverflow.com/questions/57348503/how-do-you-store-a-tuple-in-a-geopandas-geodataframe
        input_gdf['NEIGHBORS'] = input_gdf.apply(lambda row: (neighbors), axis=1)
        
    return input_gdf

def get_neighbors(input_gdf, input_id_field, input_id):
    for index, feature in input_gdf.iterrows():   
        if feature[input_id_field] == input_id:
            #print(tile.tile_num)
            # get 'not disjoint' countries
            neighbors = input_gdf[~input_gdf.geometry.disjoint(feature.geometry)][input_id_field].tolist()
            #print(feature.tile_num)
            # remove own name of the country from the list
            neighbors = [ fid for fid in neighbors if feature[input_id_field] != fid ]
            #print(neighbors)

            # add names of neighbors as NEIGHBORS value
            #boreal_tile_index.at[index, "NEIGHBORS"] = ", ".join(neighbors)
            if False:
                # This will put the neighbors list into a field in the subset df
                # https://stackoverflow.com/questions/57348503/how-do-you-store-a-tuple-in-a-geopandas-geodataframe
                subset_df = input_gdf.loc[input_gdf[input_id_field] == input_id]
                subset_df["NEIGHBORS"] = None
                subset_df['NEIGHBORS'] = subset_df.apply(lambda row: (neighbors), axis=1)

    
    return neighbors

def get_raster_zonalstats(in_gdf, vrt_fn, STATS_LIST = ['max','mean', 'median','std','min','sum','count']):
    '''For each feature in the in_gdf, append cols from STATS_LIST based on the raster summary stats for the same region in vrt_fn'''
    import rasterstats
    # Use join
    out_gdf = in_gdf.reset_index(drop=True).join(
                                                pd.DataFrame(
                                                    rasterstats.zonal_stats(
                                                        vectors=in_gdf['geometry'], 
                                                        raster=vrt_fn, 
                                                        stats=STATS_LIST
                                                    )
                                                )
                                            )
    return out_gdf

def GET_TILES_NEEDED(DPS_DATA_TYPE = 'HLS',
                    boreal_tile_index_path = '/projects/my-public-bucket/boreal_tiles_v003.gpkg',
                    GROUP_FIELD = 'tile_group',
                    tindex_master_fn = '/projects/my-private-bucket/dps_output/do_HLS_stack_3-1-2_ubuntu/master/2022/03/HLS_tindex_master.csv',#'s3://maap-ops-workspace/nathanmthomas/dps_output/do_HLS_stack_3-1-2_ubuntu/master/2022/03/HLS_tindex_master.csv',
                    topo_tindex_master_fn = '/projects/shared-buckets/nathanmthomas/DPS_tile_lists/Topo_tindex_master.csv',
                    bad_tiles = [3540,3634,3728,3823,3916,4004], #Dropping the tiles near antimeridian that reproject poorly.
                    REMOVE_BAD_TILES = False,
                    REDO_TILES_LIST = None,
                    FIND_TILE_GROUP = None
                   ):

    # Get all boreal tiles
    #shared-buckets/nathanmthomas/boreal_grid_albers90k_gpkg.gpkg
    boreal_tile_index = geopandas.read_file(boreal_tile_index_path)
    
    if REMOVE_BAD_TILES:
        # Remove bad tiles
        boreal_tile_index = boreal_tile_index[~boreal_tile_index['tile_num'].isin(bad_tiles)]

    hls_tindex_master = pd.read_csv(tindex_master_fn)
    topo_tindex_master = pd.read_csv(topo_tindex_master_fn)
    hls_tindex = boreal_tile_index.merge(hls_tindex_master[['tile_num','s3_path','local_path']], how='right', on='tile_num')
    topo_tindex = boreal_tile_index.merge(topo_tindex_master[['tile_num','s3_path','local_path']], how='right', on='tile_num')
    
    if REDO_TILES_LIST is not None:
        redo_hls_tindex = hls_tindex[hls_tindex['tile_num'].isin(REDO_TILES_LIST)]
        return redo_hls_tindex
    else:
        print(boreal_tile_index.groupby(GROUP_FIELD)[GROUP_FIELD].agg(['count']))

        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = [16, 16]

        print(f"Tile status report for {DPS_DATA_TYPE} from {tindex_master_fn}:")
        print(f'\t# of boreal tiles in boreal v003:\t\t\t{len(boreal_tile_index)}')

        # Get water tiles
        water_tiles = [] #list(set(boreal_tile_index.tile_num) - set(topo_tindex.tile_num) )
        print(f'\t# of boreal tiles in water:\t\t\t\t{len(water_tiles)}')

        NUM_STUDY_TILES = len(boreal_tile_index[~boreal_tile_index['tile_num'].isin(water_tiles)])
        print(f'\t# of boreal tiles used study (from Topo coverage):\t{NUM_STUDY_TILES}')

        ax = boreal_tile_index[~boreal_tile_index['tile_num'].isin(water_tiles)].plot(column=GROUP_FIELD, legend=True)
        #ax = tiles_topo_index.plot(color='gray', ax=ax)
        ax = hls_tindex.plot(color='black', ax = ax)
        print(f'\t# of boreal tiles with {DPS_DATA_TYPE}:\t\t\t\t{len(hls_tindex)}')

        needed_tindex = boreal_tile_index[~boreal_tile_index['tile_num'].isin(hls_tindex.tile_num.to_list() + water_tiles)]
        
        if FIND_TILE_GROUP is not None:
            needed_tindex = needed_tindex[needed_tindex[GROUP_FIELD] == FIND_TILE_GROUP]
        else:
            FIND_TILE_GROUP = 'all'
        LIST_TILES_NEEDED = needed_tindex.tile_num.to_list()
        print(f'\t# of boreal tiles still needing {DPS_DATA_TYPE} from {FIND_TILE_GROUP}:\t{len(LIST_TILES_NEEDED)}')
        # The next 100 tiles in line for processing
        #needed_tindex.iloc[0:100].plot(color='#525252', ax = ax)
        needed_tindex.plot(column=GROUP_FIELD, legend=True, ax=ax)
        
        return LIST_TILES_NEEDED
    
def BUILD_TABLE_JOBSTATUS(submit_results_df):
    import xmltodict
    job_status_df = pd.concat([pd.DataFrame(xmltodict.parse(maap.getJobStatus(job_id).content)).transpose() for job_id in submit_results_df.job_id.to_list()])
    job_status_df = submit_results_df.merge(job_status_df, how='left', left_on='job_id',  right_on='wps:JobID')
    
    print(f'Count total jobs:\t{len(job_status_df)}')
    print(f"Count pending jobs:\t{job_status_df[job_status_df['wps:Status'] =='Accepted'].shape[0]}")
    print(f"Count running jobs:\t{job_status_df[job_status_df['wps:Status'] =='Running'].shape[0]}")
    
    NUM_FAILS = job_status_df[job_status_df['wps:Status'] =='Failed'].shape[0]
    NUM_SUCCEEDS = job_status_df[job_status_df['wps:Status'] =='Succeeded'].shape[0]
    print(f"Count succeeded jobs:\t{NUM_SUCCEEDS}")
    print(f"Count failed jobs:\t{NUM_FAILS}")
    if NUM_FAILS > 0:
        print(f"% of failed jobs:\t{round(NUM_FAILS / ( NUM_FAILS + NUM_SUCCEEDS ), 4) * 100}\n")
    else:
        print(f"% of failed jobs:\tNothing has failed...yet\n")
    
    return job_status_df

def func(elem):
    '''key to sort a list based on a substring'''
    return int(elem.split('_')[-1].split('.')[0])