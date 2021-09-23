import os
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import subprocess
import argparse

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
        
    parser.add_argument("-t", "--type", type=str, choices=['Landsat', 'Topo', 'ATL08', 'all'], help="Specify the type of tiles to index from DPS output")
    parser.add_argument("-y", "--dps_year", type=str, default=2021, help="Specify the year of the DPS output")
    parser.add_argument("-o", "--outdir", type=str, default="/projects/my-public-bucket/DPS_tile_lists", help="Ouput dir for csv list of DPS'd tiles")
    parser.add_argument("--seg_str_atl08", type=str, default="_30m", help="String indicating segment length from ATL08 rebinning")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    if args.type is 'all':
        TYPE_LIST = ['Landsat', 'Topo', 'ATL08']
    else:
        TYPE_LIST = [args.type]
    
    for TYPE in TYPE_LIST:
        
        print("\nBuilding a list of tiles: ",TYPE)
        print("\nOutput dir: ", args.outdir)
        
        out_name = TYPE + "_tindex_master.csv"
        
        if "Landsat" in TYPE:
            root = f"/projects/my-private-bucket/dps_output/do_landsat_stack_3-1-2_ubuntu/ops/{args.dps_year}/"
            ends_with_str = "_dps.tif"
        if "Topo" in TYPE:
            root = f"/projects/my-private-bucket/dps_output/do_topo_stack_3-1-5_ubuntu/ops/{args.dps_year}/"
            ends_with_str = "_stack.tif"
        if "ATL08" in TYPE:
            #root = f"/projects/my-private-bucket/dps_output/run_extract_ubuntu/ops/{args.dps_year}/"
            root = f"/projects/shared-buckets/montesano/run_extract_atl08_orig_ubuntu/master/{args.dps_year}/07/14"
            ends_with_str = args.seg_str_atl08+".csv"
            
        df = pd.DataFrame(columns=[col_name, 'tile_num'])

        for dir, subdir, files in os.walk(root):
            for fname in files:
                if fname.endswith(ends_with_str): 
                    
                    tile_num = fname.split('_')[1]
                    
                    if "ATL08" in TYPE:
                        df = df.append({col_name:os.path.join(dir+"/", fname), 'tile_num':'NA'},ignore_index=True)
                    else:
                        df = df.append({col_name:os.path.join(dir+"/", fname), 'tile_num':tile_num},ignore_index=True)

        print(len(df.location.values))

        df.to_csv(os.path.join(args.outdir, out_name))

if __name__ == "__main__":
    main()