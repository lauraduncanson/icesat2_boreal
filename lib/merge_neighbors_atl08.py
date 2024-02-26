

import os
import pandas as pd
import geopandas as gpd

import argparse

def local_to_s3(url, user = 'nathanmthomas', type='public'):
    ''' A Function to convert local paths to s3 urls'''
    if type == 'public':
        replacement_str = f's3://maap-ops-workspace/shared/{user}'
    else:
        replacement_str = f's3://maap-ops-workspace/{user}'
    return url.replace(f'/projects/my-{type}-bucket', replacement_str)

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

def main():
    '''
    Script to merge ATL08 filt Geoparquets or CSVs for adjacent (neighbor) tiles
    Returns a merged Geoparquet and/or CSV of the filtered ATL08 for the focal tile and its max 8 tile neighbors
    Used for spatially adaptive model building
    This is run after DPS of covariate extraction of raster pixel values to filtered ATL08 and before mapBoreal.R
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-in_tile_num", type=int, help="The id number of an input vector tile that will define the bounds for ATL08 subset")
    parser.add_argument("-in_tile_fn",  type=str, default='/projects/shared-buckets/montesano/databank/boreal_tiles_v004.gpkg', help="The input filename of a set of vector tiles that will define the bounds for ATL08 subset")
    parser.add_argument("-in_tile_field", type=str, default='tile_num', help="The name of the field that holds the tile ids")
    parser.add_argument("-csv_list_fn", type=str, default='s3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/process_atl08_boreal/030m/2020/ATL08_filt_tindex_master.csv', help="The tindex master list of all paths to filtered + extracted covariate ATL08 Geoparquets/CSVs")
    parser.add_argument("-DPS_DATA_USER", type=str, default='montesano', help="The name for the MAAP user associated with the processing of the filtered ATL08 Geoparquet/CSV files")
    parser.add_argument("-out_dir", type=str, default=None, help='Output dir of merge neighbors Geoparquet and/or CSV for input tile id')
    
    args = parser.parse_args()
    
    in_tile_num = args.in_tile_num
    in_tile_fn = args.in_tile_fn
    in_tile_field = args.in_tile_field
    csv_list_fn = args.csv_list_fn
    DPS_DATA_USER = args.DPS_DATA_USER
    out_dir = args.out_dir
    
    # Get the vector tiles
    in_tile_gdf = gpd.read_file(in_tile_fn)
    
    in_tile_gdf[in_tile_field] = in_tile_gdf[in_tile_field].astype(int)
    
    # Get list of neighbor tiles for a given tile_num
    neighbor_tile_ids = get_neighbors(in_tile_gdf, in_tile_field, in_tile_num)
    print(f"Neighbor tile ids: {neighbor_tile_ids}")
    print(f"# of neighbor tiles: {len(neighbor_tile_ids)}")
    
    # Build up a dataframe of dps output ATL08 filtered CSVs
    ATL08_filt_tindex_master = pd.read_csv(csv_list_fn, storage_options={'anon':True})
    
    ATL08_filt_tindex_master['s3'] = [local_to_s3(local_path, user=DPS_DATA_USER, type = 'private') for local_path in ATL08_filt_tindex_master['local_path']]

    focal_atl08_gdf_fn = ATL08_filt_tindex_master['s3'].loc[ATL08_filt_tindex_master.tile_num == in_tile_num].tolist()[0]

    if out_dir is None:
        # Get the focal tile's ATL08 filt CSV name to use to make out_csv_fn
        out_dir = os.path.split(focal_atl08_gdf_fn)[0]
    
    # For neighbor tiles, get subset of ATL08 filtered CSVs as a fn list of their s3 paths assciated
    ATL08_filt_csv_s3_fn_list = [ATL08_filt_tindex_master['s3'].loc[ATL08_filt_tindex_master.tile_num == tile_id].tolist() for tile_id in neighbor_tile_ids]
    
    # THis produces a list of lists in which empty spots are removed
    ATL08_filt_csv_s3_fn_list = list(filter(None, ATL08_filt_csv_s3_fn_list))
    print(f"# of neighbor tiles with filtered ATL08 geodataframes with extracted covariates: {len(ATL08_filt_csv_s3_fn_list)} ")
    
    # Convert from list of lists to list
    ATL08_filt_csv_s3_fn_list = [item for sublist in ATL08_filt_csv_s3_fn_list for item in sublist]
    
    # Add the focal tile fn to list
    ATL08_filt_csv_s3_fn_list = [focal_atl08_gdf_fn] + ATL08_filt_csv_s3_fn_list
    
    # This handles 5-zero and 6-0 padded tile_num strings and writes to 6-0 padded strings at end of filename
    if f'_{in_tile_num:06}.' in focal_atl08_gdf_fn:
        out_fn_base_no_ext = os.path.basename(focal_atl08_gdf_fn).split(f'_{in_tile_num:06}.')[0] + f'_merge_neighbors_{in_tile_num:06}'
    else:
        out_fn_base_no_ext = os.path.basename(focal_atl08_gdf_fn).split(f'_{in_tile_num:05}.')[0] + f'_merge_neighbors_{in_tile_num:06}'

    # Update for Phase 3 uses geoparquet geodataframes of filtered ATL08
    if focal_atl08_gdf_fn.endswith('parquet'):
        input_ext='parquet'
        atl08 = pd.concat([gpd.read_parquet(f, storage_options={'anon':True}) for f in ATL08_filt_csv_s3_fn_list], sort=False)
        # Write df to Geoparquet
        #out_parquet_fn = os.path.join(out_dir, os.path.basename(focal_atl08_gdf_fn).split(f'.{input_ext}')[0] + f'_merge_neighbors_{in_tile_num:06}.parquet')
        out_parquet_fn = os.path.join(out_dir, out_fn_base_no_ext + '.parquet')
        print(f'Wrote out: {out_parquet_fn}')
        atl08.to_parquet(out_parquet_fn)
    else:
        input_ext='csv'
        # Read these ATL08 filtered CSVs into a single df
        atl08 = pd.concat([pd.read_csv(f, storage_options={'anon':True}) for f in ATL08_filt_csv_s3_fn_list], sort=False)
    
    print(f'Focal tile + neighbors shape: {atl08.shape}')
    
    # Write df to CSV (regardless of what input type was...)
    #out_csv_fn = os.path.join(out_dir, "atl08_004_30m_filt_merge_neighbors_" + str(f'{in_tile_num:05}.csv') )
    #out_csv_fn = os.path.join(out_dir, os.path.basename(focal_atl08_gdf_fn).split(f'.{input_ext}')[0] + f'_merge_neighbors_{in_tile_num:06}.csv')
    out_csv_fn = os.path.join(out_dir, out_fn_base_no_ext + '.csv')
    print(f'Wrote out: {out_csv_fn}')
    atl08.to_csv(out_csv_fn)

if __name__ == "__main__":
    main()