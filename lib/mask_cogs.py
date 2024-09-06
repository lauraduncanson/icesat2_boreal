import sys
sys.path.append('/projects/icesat2_boreal/lib')
import warnings
from multiprocessing import Pool
import pandas as pd

import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.RasterioDeprecationWarning)

from maskcog_wrapper import maskcog_wrapper, hacked_out_dest_mask_path

from mosaiclib import AGB_TINDEX_FN_DICT, LC_TINDEX_FN_DICT


def get_fn_pairs_list(tindex_src_fn, tindex_mask_fn, path_col, FOCAL_TILE=None):
    
    list_fn_pairs = []
    tindex_src = pd.read_csv(tindex_src_fn)
    tindex_mask = pd.read_csv(tindex_mask_fn)
    
    if FOCAL_TILE is not None:
        TILE_LIST = [FOCAL_TILE]
    else:
        TILE_LIST = tindex_src.tile_num.to_list()
        
    for FOCAL_TILE in TILE_LIST:
        src_fn = tindex_src[tindex_src.tile_num == FOCAL_TILE][path_col].to_list()[0]
        mask_fn = tindex_mask[tindex_mask.tile_num == FOCAL_TILE][path_col].to_list()[0]
        list_fn_pairs.append((src_fn, mask_fn))
    
    return list_fn_pairs



if __name__ == '__main__':
    # for the full run, we just don't pass a FOCAL_TILE

    list_of_fn_pairs = get_fn_pairs_list(
        AGB_TINDEX_FN_DICT['2020_v2.0'],
        LC_TINDEX_FN_DICT['c2020updated'],
        's3_path', #'local_path', # here we specify the local_path column instead of the s3_path column
        #FOCAL_TILE=3375
    )[:2]
    print(list_of_fn_pairs)
    print(hacked_out_dest_mask_path(list_of_fn_pairs[0][0]))
    print(hacked_out_dest_mask_path(list_of_fn_pairs[1][0]))

    with Pool(processes=4) as pool:
        returned_stuff = pool.map(maskcog_wrapper, list_of_fn_pairs)

    results = pd.DataFrame(returned_stuff, columns=['tindex', 'status'])
    results.to_csv("/projects/my-public-bucket/local_output/maskcog_job_status.csv")
    print(f"{results[results['status'] == 'SUCCESS'].shape[0]}/ {results.shape[0]} succeded.")
    print(results.head())
