import os
import build_stack 
import logging

logging.basicConfig(
    filename='/projects/my-public-bucket/local_output/mask_cog_logs.txt',
    level=logging.INFO
)
logging.info("masking begins here...")

def hacked_out_dest_mask_path(biomass_fn):
    local_biomass_fn = biomass_fn.replace(
        's3://maap-ops-workspace/lduncanson',
        '/projects/my-private-bucket'
    )
    local_biomass_fn_base = os.path.basename(local_biomass_fn)
    root, ext = os.path.splitext(local_biomass_fn)

    mask_fn = root + '_masked' + ext
    return mask_fn

def maskcog_wrapper(paths):
    biomass_fn, _ = paths
    out_cog_fn = hacked_out_dest_mask_path(biomass_fn)
    tindex = os.path.basename(out_cog_fn).split('_')[-2]
    try:
        build_stack.mask_cog(paths, mask_val_list=[60], overwrite=False, out_cog_fn=out_cog_fn)
        logging.info(f"PASS: masked {paths[0]} using {paths[1]} into {out_cog_fn}")
        return (tindex, 'SUCCESS')
    except Exception as e:
        logging.error(f"FAIL: could not mask {paths[0]}. Error: {e}")
        return (tindex, 'ERROR')

if __name__ == '__main__':
    f1 = 's3://maap-ops-workspace/lduncanson/dps_output/run_boreal_biomass_map/boreal_agb_2024_v6/AGB_H30_2020/Version2_SD/2024/07/17/10/30/12/581227/boreal_agb_202407171721237382_004308.tif'
    f1_local_masked = '/projects/my-private-bucket/dps_output/run_boreal_biomass_map/boreal_agb_2024_v6/AGB_H30_2020/Version2_SD/2024/07/17/10/30/12/581227/boreal_agb_202407171721237382_004308_masked.tif'
    f2 = 's3://maap-ops-workspace/lduncanson/dps_output/run_boreal_biomass_map/boreal_agb_2024_v6/AGB_H30_2020/Version2_SD/2024/07/17/10/30/20/762418/boreal_agb_202407171721237320_041437.tif'
    f2_local_masked = '/projects/my-private-bucket/dps_output/run_boreal_biomass_map/boreal_agb_2024_v6/AGB_H30_2020/Version2_SD/2024/07/17/10/30/20/762418/boreal_agb_202407171721237320_041437_masked.tif'
    assert hacked_out_dest_mask_path(f1) == f1_local_masked, 'FAIL'
    assert hacked_out_dest_mask_path(f2) == f2_local_masked, 'FAIL'
    print('tests PASSED!')
