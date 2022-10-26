import os
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import subprocess
import argparse
import s3fs

try:
    from maap.maap import MAAP
    HAS_MAAP = True
except ImportError:
    print('NASA MAAP is unavailable')
    HAS_MAAP = False

def local_to_s3(url, user = 'nathanmthomas', type='public'):
    ''' A Function to convert local paths to s3 urls'''
    if type == 'public':
        replacement_str = f's3://maap-ops-workspace/shared/{user}'
    else:
        replacement_str = f's3://maap-ops-workspace/{user}'
    return url.replace(f'/projects/my-{type}-bucket', replacement_str)

def s3fs_to_https(url, user = 'nathanmthomas', type='public'):
    ''' A Function to convert s3fs urls to https'''
    return url.replace("maap-ops-workspace", "https://maap-ops-workspace.s3.amazonaws.com")

def s3_to_local(url, user = 'nathanmthomas', type='private'):
    ''' A Function to convert s3fs urls to local paths of private bucket'''
    if type == 'public':
        cur_str = f's3://maap-ops-workspace/shared/{user}'
        replace_str = f'/projects/my-public-bucket/shared/{user}'
    else:
        cur_str = f's3://maap-ops-workspace/{user}'
        replace_str = f'/projects/my-private-bucket'
    return url.replace(cur_str, replace_str)

# This is the glob.glob approach; the other is the os.walk approach
def get_atl08_csv_list(dps_dir_csv, seg_str, csv_list_fn, col_name='local_path'):
    print(dps_dir_csv + "/**/ATL08*" + seg_str + ".csv") 
    #seg_str="_30m"
    print('Running glob.glob to return a list of csv paths...')
    all_atl08_csvs = glob.glob(dps_dir_csv + "/**/ATL08*" + seg_str + ".csv", recursive=True)
    print(len(all_atl08_csvs))
    all_atl08_csvs_df = pd.DataFrame({col_name: all_atl08_csvs})
    all_atl08_csvs_df.to_csv(csv_list_fn)
    return(all_atl08_csvs_df)

def handle_duplicates(df, FOCAL_FIELD_NAME, TYPE: str, RETURN_DUPS):
        ######
        # Handle duplicate tiles (or files, if ATL08) by sorting and keeping the latest
        
        # Sort (descending; newest first)
        df.sort_values(by=['local_path'], ascending=False, inplace=True)
                
        # Check
        dropped  = df[df.duplicated(subset=[FOCAL_FIELD_NAME], keep='first')]
        if len(dropped) > 0:
            dropped.loc[:,'status'] = 'dropped'
            # TODO: These arent handled correctly at the moment. 
            retained = df[df.duplicated(subset=[FOCAL_FIELD_NAME], keep='last')]
            retained.loc[:,'status'] = 'retained'
            if RETURN_DUPS:
                #pd.concat([dropped, retained]).to_csv(os.path.splitext(out_tindex_fn)[0] +  '_duplicates.csv')
                dropped.to_csv(os.path.splitext(out_tindex_fn)[0] +  '_duplicates.csv')
                # dropped['local_path'].to_list()[0:10]
                # retained['local_path'].to_list()[0:10]
        else:
            print('\nNo duplicates found.\n')
        
        # Drop duplicates, keeping the latest version of the tile
        df = df.drop_duplicates(subset=[FOCAL_FIELD_NAME], keep='first')
        
        if FOCAL_FIELD_NAME == 'tile_num':
            # Make sure the tile_num field is int (not object)
            df['tile_num'] = df['tile_num'].astype(str).astype(int)
        
        num_without_duplicates = df.shape[0]
        print(f"# of duplicate tiles: {dropped.shape[0]}")
        print(f"Final # of tiles: {num_without_duplicates}")
        print(f"df shape : {df.head()}")
        return df

def main():
    """
    Build a list of output geotiff tiles returned from DPS jobs
    This list is used to build a MosaicJSON for display in 3-DPSmosaic.ipynb
    
    Example calls: 
        python build_tindex_master.py -t Topo -y 2021 --outdir /projects/my-public-bucket/DPS_tile_lists
        python build_tindex_master.py -t Landsat -y 2021
    """
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument("-t", "--type", type=str, choices=['LC','HLS','Landsat', 'Topo', 'ATL08', 'ATL08_filt', 'AGB', 'all'], help="Specify the type of tiles to index from DPS output")
    parser.add_argument("-y", "--dps_year", type=str, default=2022, help="Specify the year of the DPS output")
    parser.add_argument("-m", "--dps_month", type=str, default=None, help="Specify the start month of the DPS output as a zero-padded string")
    parser.add_argument("-m_list", "--dps_month_list", nargs='+', type=str, default=None, help="Specify the list of month of the DPS output as a zero-padded string")
    parser.add_argument("-d_min", "--dps_day_min", type=int, default=1, help = "Specify the first day of the DPS output")
    parser.add_argument("-d_max", "--dps_day_max", type=int, default=31, help="")
    parser.add_argument("-alg_name", type=str, choices=['do_HLS_stack_3-1-2_ubuntu','do_landsat_stack_3-1-2_ubuntu',
                                                        'do_topo_stack_3-1-5_ubuntu','run_extract_filter_atl08_ubuntu',
                                                        'run_tile_atl08_ubuntu','run_boreal_biomass_v5_ubuntu','run_boreal_biomass_quick_ubuntu','run_build_stack_ubuntu'], 
                        default='run_boreal_biomass_v5_ubuntu', help="The MAAP algorithm name used to produce output for the tindex")
    parser.add_argument("--maap_version", type=str, default='master', help="The version of MAAP")
    parser.add_argument("-o", "--outdir", type=str, default="/projects/my-public-bucket/DPS_tile_lists", help="Ouput dir for csv list of DPS'd tiles")
    parser.add_argument("--seg_str_atl08", type=str, default="_30m", help="String indicating segment length from ATL08 rebinning")
    parser.add_argument("-s", "--ends_with_str", type=str, default=".tif", help="String indicating ending pattern of files of interest.")
    parser.add_argument("-b","--bucket_name", type=str, default=None, help="s3 bucket name for file searching.")
    parser.add_argument("-r", "--root_key", type=str, default=None, help="Root dir for data search")
    parser.add_argument("--col_name", type=str, default="s3_path", help="Column name for the local path of the found files")
    parser.add_argument('--DEBUG', dest='DEBUG', action='store_true', help='Do debugging')
    parser.set_defaults(DEBUG=False)
    parser.add_argument('-local_dir', type=str, default=None, help='Local testing dir')
    parser.add_argument('--LOCAL_TEST', dest='LOCAL_TEST', action='store_true', help='Do local testing')
    parser.set_defaults(LOCAL_TEST=False)
    parser.add_argument('--RETURN_DUPS', dest='RETURN_DUPS', action='store_true', help='Return a df of dropped and a df of retained duplicate tiles')
    parser.set_defaults(RETURN_DUPS=False)
    parser.add_argument('--tindex_append', dest='tindex_append', action='store_true', help='Append data frame to existing tindex master csv')
    parser.set_defaults(tindex_append=False)
    args = parser.parse_args()
    
    s3 = s3fs.S3FileSystem(anon=True)
    
    if HAS_MAAP:
        bucket = "s3://maap-ops-workspace"
    else:
        if args.bucket_name is None:
            print("MUST SPECIFY a s3 BUCKET DIR (-b) if not on MAAP")
            os._exit(1)
        else:
            bucket = "s3://" + args.bucket_name
            #bucket = args.bucket_name
    
    col_name = args.col_name
    DEBUG = args.DEBUG
    dps_month = args.dps_month
    dps_month_list = args.dps_month_list
    alg_name = args.alg_name
    
    if HAS_MAAP and dps_month is None and dps_month_list is None:
        print('You need to specify either a -dps_month or a -dps_month_list')
        os._exit(1)
    
    if dps_month_list is None:
        dps_month_list = [dps_month]

    if not 's3://' in args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
    
    if args.type == 'all':
        TYPE_LIST = ['Landsat', 'Topo', 'ATL08', 'ATL08_filt', 'AGB','HLS','LC']
    else:
        TYPE_LIST = [args.type]
    
    for TYPE in TYPE_LIST:
        
        print("\nBuilding a list of tiles:")
        print(f"MAAP version:\t\t{args.maap_version}")
        print(f"Type:\t\t{TYPE}")
        print(f"Year:\t\t{args.dps_year}")
        print(f"Month:\t\t{dps_month_list}")
        print(f"Days:\t\t{args.dps_day_min}-{args.dps_day_max}")
        print("\nOutput dir: ", args.outdir)
        
        out_name = TYPE + "_tindex_master.csv"
        out_tindex_fn = os.path.join(args.outdir, out_name)
        
        str_exclude_list = ['SAMPLE', 'checkpoint']
        
        if HAS_MAAP:
            
            if "LC" in TYPE:
                user = 'nathanmthomas'
                dps_out_searchkey_list = [f"{user}/dps_output/{alg_name}/{args.maap_version}/{args.dps_year}/{dps_month}/{format(d, '02')}/**/*.tif" for d in range(args.dps_day_min, args.dps_day_max + 1) for dps_month in dps_month_list]
                ends_with_str = "_cog.tif"
        
            if "HLS" in TYPE:
                user = 'nathanmthomas'
                dps_out_searchkey_list = [f"{user}/dps_output/{alg_name}/{args.maap_version}/{args.dps_year}/{dps_month}/{format(d, '02')}/**/*.tif" for d in range(args.dps_day_min, args.dps_day_max + 1) for dps_month in dps_month_list]
                ends_with_str = "_dps.tif"
            if "Landsat" in TYPE:
                user = 'nathanmthomas'
                dps_out_searchkey_list = [f"{user}/dps_output/{alg_name}/{args.maap_version}/{args.dps_year}/{dps_month}/{format(d, '02')}/**/*_dps.tif" for d in range(args.dps_day_min, args.dps_day_max + 1) for dps_month in dps_month_list]
                ends_with_str = "_dps.tif"
            if "Topo" in TYPE:
                user = 'nathanmthomas'
                dps_out_searchkey_list = [f"{user}/dps_output/{alg_name}/{args.maap_version}/{args.dps_year}/{dps_month}/{format(d, '02')}/**/*_stack.tif" for d in range(args.dps_day_min, args.dps_day_max + 1) for dps_month in dps_month_list]
                ends_with_str = "_stack.tif"
            if "ATL08" in TYPE:
                user = 'lduncanson'
                dps_out_searchkey_list = [f"{user}/dps_output/{alg_name}/{args.maap_version}/{args.dps_year}/{dps_month}/{format(d, '02')}/**/*{args.seg_str_atl08}.csv" for d in range(args.dps_day_min, args.dps_day_max + 1) for dps_month in dps_month_list]
                ends_with_str = args.seg_str_atl08+".csv"
            if "filt" in TYPE:
                user = 'lduncanson'
                dps_out_searchkey_list = [f"{user}/dps_output/{alg_name}/{args.maap_version}/{args.dps_year}/{dps_month}/{format(d, '02')}/**/*.csv" for d in range(args.dps_day_min, args.dps_day_max + 1) for dps_month in dps_month_list]
                ends_with_str = ".csv"
            if "AGB" in TYPE:
                user = 'lduncanson'
                dps_out_searchkey_list = [f"{user}/dps_output/{alg_name}/{args.maap_version}/{args.dps_year}/{dps_month}/{format(d, '02')}/**/*.tif" for d in range(args.dps_day_min, args.dps_day_max + 1) for dps_month in dps_month_list]
                ends_with_str = ".tif"
                
        else:
            if args.root_key is None:
                print("MUST SPECIFY ROOT DIR (-r) if not on MAAP")
                os._exit(1)
            else:
                dps_out_searchkey_list = [f"{args.root_key}/**/*{args.ends_with_str}"]
                [print(f"bucket, searchkey: {os.path.join(bucket, searchkey)}") for searchkey in dps_out_searchkey_list]
                
        if args.LOCAL_TEST:
            print('LOCAL TEST')
            dps_out_searchkey_list = [f"{args.local_dir}/*{args.ends_with_str}"]
            
        if args.DEBUG:
            print(f'bucket = {bucket}')
            print(f'dps_out_searchkey_list = {dps_out_searchkey_list}')
            print(f'col_name = {col_name}')
            print('\nDF call with search:\ndf = pd.concat([pd.DataFrame(s3.glob(os.path.join(bucket, searchkey)), columns=[col_name]) for searchkey in dps_out_searchkey_list])')
        
        df = pd.concat([pd.DataFrame(s3.glob(os.path.join(bucket, searchkey)), columns=[col_name]) for searchkey in dps_out_searchkey_list])
        #print(df.head())
        
        if len(df) == 0:
            print('Nothing found. Check year and month. Exiting.')
            os._exit(1)
        
        # Remove rows of files that shouldnt be here
        #df = pd.concat([df[~df[col_name].str.contains(string_exclude)] for string_exclude in str_exclude_list])
        
        for string_exclude in str_exclude_list:
            df = df[~df['s3_path'].str.contains(string_exclude)] 
        
        df['s3_path'] = [f's3://{f}' for f in df[col_name].tolist()]
        df['local_path'] = [s3_to_local(f, user=user) for f in df[col_name].tolist()]
        df['file'] = [os.path.basename(f) for f in df[col_name].tolist()]
        print(df.head()) 
        
        # Get the tile num from the file string, which is in different places
        if 'LC' in TYPE:
            df['tile_num'] = df['file'].str.split('_', expand=True)[4].str.strip('*.tif')
        if 'AGB' in TYPE:
            df['tile_num'] = df['file'].str.split('_', expand=True)[3].str.strip('*.tif')
        if 'Topo' in TYPE:
            df['tile_num'] = df['file'].str.split('_', expand=True)[1].str.strip('*.tif')
        if 'Landsat' in TYPE:
            df['tile_num'] = df['file'].str.split('_', expand=True)[6].str.strip('*.tif')
        if 'HLS'in TYPE:
            df['tile_num'] = df['file'].str.split('_', expand=True)[1].str.strip('*.tif') 
        if 'ATL08' in TYPE:
            
            if 'ATL08_filt' in TYPE:

                df['tile_num'] = df['file'].str.split('_', expand = True)[7].str.split('.csv', expand = True)[0]
                
                # Get n_obs column for every tile, and join on tile_num
                tile_num_list = [os.path.basename(f).split('_')[7].split('.csv')[0] for f in df[col_name].tolist()]
                n_obs_list = [pd.read_csv(f).shape[0] for f in df[col_name].to_list()]
                df_nobs = pd.DataFrame(data={'tile_num': tile_num_list, 'n_obs': n_obs_list})
                df = df.join(df_nobs[['tile_num','n_obs']].set_index('tile_num'), how='left', on='tile_num')
            else:
                df['tile_num'] =  'NA'
                    
        num_with_duplicates = df.shape[0]
        
        if args.tindex_append:
            print(f'Appending to existing tindex...')
            df_existing = pd.read_csv(out_tindex_fn)
            df = df_existing.append(df)
            
        if TYPE == 'ATL08':
            df = handle_duplicates(df, 'file', TYPE, args.RETURN_DUPS)
        else:
            df = handle_duplicates(df, 'tile_num', TYPE, args.RETURN_DUPS) 
        
        print(f'Writing tindex master csv: {out_tindex_fn}')
        df.to_csv(out_tindex_fn)
        return df
if __name__ == "__main__":
    main()