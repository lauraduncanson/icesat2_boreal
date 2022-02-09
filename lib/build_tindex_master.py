import os
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import subprocess
import argparse

def local_to_s3(url, user = 'nathanmthomas', type='public'):
    ''' A Function to convert local paths to s3 urls'''
    if type == 'public':
        replacement_str = f's3://maap-ops-workspace/shared/{user}'
    else:
        replacement_str = f's3://maap-ops-workspace/{user}'
    return url.replace(f'/projects/my-{type}-bucket', replacement_str)

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

def main():
    """
    Build a list of output geotiff tiles returned from DPS jobs
    This list is used to build a MosaicJSON for display in 3-DPSmosaic.ipynb
    
    Example calls: 
        python CountOutput.py -t Topo -y 2021 --outdir /projects/my-public-bucket/DPS_tile_lists
        python CountOutput.py -t Landsat -y 2021
    """
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument("-t", "--type", type=str, choices=['Landsat', 'Topo', 'ATL08', 'ATL08_filt', 'AGB', 'all'], help="Specify the type of tiles to index from DPS output")
    parser.add_argument("-y", "--dps_year", type=str, default=2022, help="Specify the year of the DPS output")
    parser.add_argument("-m", "--dps_month", type=int, default=9, help="Specify the month of the DPS output as a zero-padded string")
    parser.add_argument("-d_min", "--dps_day_min", type=int, default=1, help = "Specify the first day of the DPS output")
    parser.add_argument("-d_max", "--dps_day_max", type=int, default=31, help="")
    parser.add_argument("-o", "--outdir", type=str, default="/projects/my-public-bucket/DPS_tile_lists", help="Ouput dir for csv list of DPS'd tiles")
    parser.add_argument("--seg_str_atl08", type=str, default="_30m", help="String indicating segment length from ATL08 rebinning")
    parser.add_argument("--col_name", type=str, default="local_path", help="Column name for the local path of the found files")
    parser.add_argument('--DEBUG', dest='DEBUG', action='store_true', help='Do debugging')
    parser.set_defaults(DEBUG=False)
    args = parser.parse_args()
    
    col_name = args.col_name
    DEBUG = args.DEBUG
    dps_month = args.dps_month
    #dps_day_min = args.dps_day_min
    #dps_day_max = args.dps_day_max
    
    #dps_day_min = format(dps_day_min, '02')
    #dps_day_max = format(dps_day_max, '02')
    dps_month = format(dps_month, '02')
    #[format(n, '03') for n in list(range(0,5))]
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    if args.type == 'all':
        TYPE_LIST = ['Landsat', 'Topo', 'ATL08', 'ATL08_filt', 'AGB']
    else:
        TYPE_LIST = [args.type]
    
    for TYPE in TYPE_LIST:
        
        print("\nBuilding a list of tiles: ",TYPE)
        print("\nOutput dir: ", args.outdir)
        
        out_name = TYPE + "_tindex_master.csv"
        str_exclude = 'xxx'
        
        if "Landsat" in TYPE:
            #dps_out_subdir = f"do_landsat_stack_3-1-2_ubuntu/ops/{args.dps_year}/"
            dps_out_subdir_list = [f"do_landsat_stack_3-1-2_ubuntu/ops/{args.dps_year}/{dps_month}/{format(d, '02')}/" for d in range(args.dps_day_min, args.dps_day_max)]
            user = 'nathanmthomas'
            ends_with_str = "_dps.tif"
        if "Topo" in TYPE:
            #dps_out_subdir = f"do_topo_stack_3-1-5_ubuntu/ops/{args.dps_year}/"
            dps_out_subdir_list = [f"do_topo_stack_3-1-5_ubuntu/ops/{args.dps_year}/{dps_month}/{format(d, '02')}/" for d in range(args.dps_day_min, args.dps_day_max)]
            user = 'nathanmthomas'
            ends_with_str = "_stack.tif"
        if "ATL08" in TYPE:
            #dps_out_subdir = f"run_extract_atl08_ubuntu/master/{args.dps_year}/{dps_month}/{dps_day_min}/"
            dps_out_subdir_list = [f"run_extract_atl08_ubuntu/master/{args.dps_year}/{dps_month}/{format(d, '02')}/" for d in range(args.dps_day_min, args.dps_day_max)]
            user = 'lduncanson'
            ends_with_str = args.seg_str_atl08+".csv"
        if "filt" in TYPE:
            #dps_out_subdir = f"run_tile_atl08_ubuntu/master/{args.dps_year}/{dps_month}/{dps_day_min}/"
            dps_out_subdir_list = [f"run_tile_atl08_ubuntu/master/{args.dps_year}/{dps_month}/{format(d, '02')}/" for d in range(args.dps_day_min, args.dps_day_max)]
            user = 'lduncanson'
            ends_with_str = ".csv"
            str_exclude = 'SAMPLE'
        if "AGB" in TYPE:
            #dps_out_subdir = f"run_boreal_biomass_ubuntu/master/{args.dps_year}/{dps_month}/{dps_day_min}/"
            dps_out_subdir_list = [f"run_boreal_biomass_ubuntu/master/{args.dps_year}/{dps_month}/{format(d, '02')}/" for d in range(args.dps_day_min, args.dps_day_max)]
            user = 'lduncanson'
            ends_with_str = ".tif"
            
        df = pd.DataFrame(columns=[col_name, 'tile_num'])

        for dps_out_subdir in dps_out_subdir_list:
        
            # Convert local root to s3
            root = '/projects/my-private-bucket/dps_output/' + dps_out_subdir
            #root = local_to_s3(root, user=user, type='private')
            print(f'Root dir: {root}')
            
            # Start count at root level 
            count_dps_out_subdir = 0
            for dir, subdir, files in os.walk(root):
                
                # Start count at root level 
                #count_dps_out_subdir = 0
                
                for fname in files:
                    
                    if fname.endswith(ends_with_str) and not str_exclude in fname:
                        
                        count_dps_out_subdir = count_dps_out_subdir + 1
                        
                        if DEBUG: print(fname)
                        
                        tile_num = fname.split('_')[1]

                        if 'AGB' in TYPE:
                            tile_num = fname.split('_')[3]

                        #if DEBUG: print(f'Tile num: {tile_num}')

                        if "ATL08" in TYPE and not "filt" in TYPE:
                            df = df.append({col_name:os.path.join(dir+"/", fname), 'tile_num':'NA'},ignore_index=True)
                        elif "ATL08_filt" in TYPE:
                            if len(fname.split('checkpoint')) > 1:
                                continue
                            print(fname)
                            tile_num = int(os.path.splitext(fname)[0].split("_")[-1])
                            df = df.append({col_name:os.path.join(dir+"/", fname), 'tile_num':tile_num},ignore_index=True)
                        else:
                            df = df.append({col_name:os.path.join(dir+"/", fname), 'tile_num':tile_num},ignore_index=True)
                        #if DEBUG: print(os.path.join(dir+"/", fname))
                        
            print(f"{count_dps_out_subdir} AGB tiles in dps_output subdir: {dps_out_subdir}")
                    
        num_with_duplicates = len(df[col_name].values)
        print(num_with_duplicates)
        
        # Drop duplicates
        df = df.drop_duplicates(subset=['tile_num'], keep='last')
        num_without_duplicates = len(df[col_name].values)
        print(f"# of duplicate tiles: {num_with_duplicates-num_without_duplicates}")
        print(f"Final # of tiles: {num_without_duplicates}")
        
        out_tindex_fn = os.path.join(args.outdir, out_name)
        print(f'Writing tindex master csv: {out_tindex_fn}')
        df.to_csv(out_tindex_fn)

if __name__ == "__main__":
    main()