import os

######
######
###### LIBRARY OF MASTER TINDEX & MOSAIC JSON files associated with various iterations of the ABoVE Boreal AGB Density maps
######
###### variables hold the paths to the tindex csv & mosaic json files on MAAP
###### import this into other notebooks: from mosaiclib import *

boreal_tile_index_path = '/projects/shared-buckets/montesano/databank/boreal_tiles_v004.gpkg'
MINI_DATELINE_TILES  = [400400,391600,382300,372800,363400,354000,4199500,4180700,4161900]
LARGE_DATELINE_TILES = [3540,3634,3728,3823,3916,4004,41995,41807,41619]
MERIDIAN_TILES       = [22938, 23219, 23828, 24109, 23548, 23782, 24670, 24389, 24108, 23501, 23547]

######
###### Titiler mosaic registrations
######
TITILER_MOSAIC_REG_DICT = {
    'S1':
        {'2019 HV summer': '90e85cea-26b0-4f9a-844c-81864c8df7a9',
         '2020 HV summer': '7422ac5e-d93d-4def-b31a-2c069b519d34',
        },
    'LC':
        {'c2020updated':'bc22b016-9cf2-46e1-bbbe-3da41a0b821a',
        },
    'TOPO':
        {'c2020updated_v2':'2b7aac49-7248-43d0-9c6c-a7ecb0c08ace',
        },
    'HLS NDVI':  
        {'2016':'124b68df-17de-48e3-8d51-8ec3a9067f74' ,
         '2017':None,
         '2018':None,
         '2018_test':None,
         '2019':None,
         '2020':'48282962-c503-42a0-b464-3811879daa15',
         '2021':None,
         '2022':None,
         '2023':None,
         '2024':None, #'8a53fc64-37fb-4c12-8949-4b0c42a8a1db'
        },
    'HLS NBR2':  
        {'2016':None ,
         '2017':None,
         '2018':None,
         '2018_test':None,
         '2019':None,
         '2020':None,
         '2021':None,
         '2022':None,
         '2023':None,
         '2024':None
        },
    'AGB':{
        '2020_v2.0': 'ea5bff1b-a5bf-497b-a01c-5dda868cb499', #'2cc6a704-255f-4e3f-a36b-5b278d24296d',
        #'2020_v2.0_masked': '8aa09548-e68a-4974-8238-6f576a2f6e31',
        '2020_v2.1': None,                                 # Ali runs w/o moss/lichen mask
        '2020_v2.2': None,                                 # Ali runs w/o moss/lichen mask w/ S1
    },
    'HT':{
        '2020_v2.0':'ddd273a5-7979-41d1-a282-62c44ded9147',# Version2_SD runs w/ moss/lichen mask
        '2020_v2.1_no_uncert': None,                        # Ali runs w/o moss/lichen mask and no uncerts
        '2020_v2.1': None,                                 # Ali runs w/o moss/lichen mask
        '2020_v2.2': 'd3985ed1-99df-422a-81e4-a9e27d8119fc' # Ali runs w/o moss/lichen mask w/ S1
    },
    'TCC':{
        '2020':None,
    },
    'TCCTREND':{
        '2020': None, #'fbfa85ec-1bbc-4870-9a3f-ca985860088a'
    },
    'AGE':{
        '2020': None,
    },
    'FORESTAGE100m': {
        '2020': None,
    },
    'FORESTAGE': {
        '2020': None,
    },
}

######
###### Boreal ATL08 granules list (from PhoReal)
######
ATL08_GRANULE_TINDEX_FN_DICT = {
    # ATL08 csv tindex for spring2022 ATL08_tindex_master_fn
    'c2020spring2022' : 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/ATL08_tindex_master.csv',
    # ATL08 csv tindex for fall2022 v1
    'c2020fall2022v1' : 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/with_gedi_rh/ATL08_tindex_master.csv',
    # ATL08 csv tindex for fall2022 v2
    'c2020fall2022v2' : 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/with_atl03_rh/ATL08_tindex_master.csv'
}

######
###### Boreal filtered ATL08 geodataframes (stored as CSVs) that have extracted covar pixel values
######
#ATL08_filt_tindex_master_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/ATL08_filt_tindex_master.csv'
#ATL08_filt_tindex_master_fn = 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/with_atl03_rh/ATL08_filt_tindex_master.csv'
#ATL08_filt_tindex_json_fn= "/projects/my-public-bucket/DPS_tile_lists/ATL08_filt_tindex_master.json"
ATL08_FILT_EXTRACT_TINDEX_FN_DICT = {
    # Final ATL08_filt - NO dateline tiles
    'c2020spring2022' : 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/ATL08_filt/c2020/tile_atl08/ATL08_filt_tindex_master.csv', # note: in nathan's now
    'c2020fall2022v1' : 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/with_gedi_rh/ATL08_filt_tindex_master.csv',
    'c2020fall2022v2' : 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/with_atl03_rh/ATL08_filt_tindex_master.csv',
    # in Phase 3 with same ATL08 v005 - ONLY dateline tiles
    'c2020_v005'      : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/tile_atl08/c2020_v005/ATL08_filt_tindex_master.csv',
    # These used TOPO_TINDEX_FN_DICT['c2020updated']
    '2019_old'        : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars_old/2019/ATL08_filt_extract_tindex_master.csv', # topo + HLS H30 + S1
    '2020_old'        : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars_old/2020/ATL08_filt_extract_tindex_master.csv', # topo + HLS H30 + S1 * agg tile 10 has missing S1 subtiles that should be fixed - not priority
    '2021_old'        : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars_old/2021/ATL08_filt_extract_tindex_master.csv', # waiting for S1 subtile tranfer from GEE ...
    '2022_old'        : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars_old/2022/ATL08_filt_extract_tindex_master.csv', # topo + HLS H30 -- could add S1, but very incomplete mosaic for this year
    '2023_old'        : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars_old/2023/ATL08_filt_extract_tindex_master.csv', # topo + HLS H30 -- could add S1, but very incomplete mosaic for this year
    # These used TOPO_TINDEX_FN_DICT['c2020updated_v2']
    '2019'            : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars/2019/ATL08_filt_extract_tindex_master.csv', # topo + HLS H30 + S1
    '2020'            : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars/2020/ATL08_filt_extract_tindex_master.csv', # topo + HLS H30 + S1 * agg tile 10 has missing S1 subtiles that should be fixed - not priority
    '2021'            : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars/2021/ATL08_filt_extract_tindex_master.csv', # waiting for S1 subtile tranfer from GEE ...
    '2022'            : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars/2022/ATL08_filt_extract_tindex_master.csv', # topo + HLS H30 -- could add S1, but very incomplete mosaic for this year
    '2023'            : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/ATL08/extract_atl08_covars/2023/ATL08_filt_extract_tindex_master.csv', # topo + HLS H30 -- could add S1, but very incomplete mosaic for this year
}

######
###### Boreal AGB mosaics
######
# Final AGB - these COGs are no longer available in dps_output - only on ORNL DAAC
AGB_c2020_noground_tindex_master_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/c2020/map_boreal_2022_rh_noground_v4/AGB_tindex_master.csv'
AGB_c2020_noground_mosaic_json_fn   = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/c2020/map_boreal_2022_rh_noground_v4/AGB_tindex_master_mosaic.json'

# c2020 Reproduction of the ORNL DAAC map but:
# no global sample contribution
# has center tile
AGB_localfull_mosaic_json_fn =          's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/BOREAL_MAP/boreal_agb_2023_v3/AGB_L30_2020/local_training_full/AGB_tindex_master_mosaic.json'

AGB_norway_mosaic_json_fn =             's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/BOREAL_MAP/boreal_agb_2023_v2/AGB_H30_2020/norway_val/AGB_tindex_master_mosaic.json'
AGB_norway50perc_mosaic_json_fn =       's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/BOREAL_MAP/boreal_agb_2023_v2/AGB_H30_2020/norway_val_loc_training_perc50/AGB_tindex_master_mosaic.json'
AGB_local_quicktest_mosaic_json_fn =    's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/BOREAL_MAP/boreal_agb_2023_v3/AGB_H30_2020/local_training_quicktest/AGB_tindex_master_mosaic.json'
AGB_H30_localtrainfull_mosaic_json_fn = 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/BOREAL_MAP/boreal_agb_2023_v3/AGB_H30_2020/local_training_full/AGB_tindex_master_mosaic.json'

AGB_prelim_tindex_master_fn =           's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/AGB_tindex_master.csv'
AGB_prelim_mosaic_json_fn =             's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/AGB_tindex_master_mosaic.json'

AGB_v2_tindex_master_fn =               's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/v2/AGB_tindex_master.csv'  # <-- 07 indicates the summer update
AGB_v2_mosaic_json_fn   =               's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/v2/AGB_tindex_master_mosaic.json'

AGB_spring2022_tindex_master_fn =       's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/06/AGB_tindex_master.csv'  # <-- 07 indicates the summer update
AGB_spring2022_mosaic_json_fn   =       's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/06/AGB_tindex_master_mosaic.json'

AGB_summer2022_tindex_master_fn =       's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/07/AGB_tindex_master.csv'  # <-- 07 indicates the summer update
AGB_summer2022_mosaic_json_fn   =       's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/07/AGB_tindex_master_mosaic.json'

AGB_fall2022_tindex_master_fn =         's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/fall2022/map_boreal_2022_v3/11/AGB_tindex_master.csv'
AGB_fall2022_mosaic_json_fn =           's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/fall2022/map_boreal_2022_v3/11/AGB_tindex_master_mosaic.json'

AGB_fall2022_noground_tindex_master_fn ='s3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/fall2022/with_atl03_rh/map_boreal_2022_rh_noground_v1/12/AGB_tindex_master.csv'
AGB_fall2022_noground_mosaic_json_fn =  's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/fall2022/with_atl03_rh/map_boreal_2022_rh_noground_v1/12/AGB_tindex_master_mosaic.json'

#AGB_winter2023_noground_tindex_master_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/winter2023/map_boreal_2022_rh_noground_v1/AGB_tindex_master.csv'
#AGB_winter2023_noground_mosaic_json_fn   = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/winter2023/map_boreal_2022_rh_noground_v1/AGB_tindex_master_mosaic.json'

AGB_TEST_mosaic_json_fn =               's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/2023/AGB_tindex_master_mosaic.json'
AGB_test_norway_mosaic_json_fn =        's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/winter2023/map_boreal_2022_rh_noground_v4/AGB_tindex_master_mosaic.json'

AGB_winter2023_noground_tindex_master_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/winter2023/map_boreal_2022_rh_noground_v4/AGB_tindex_master.csv'
AGB_winter2023_noground_mosaic_json_fn   = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/AGB/winter2023/map_boreal_2022_rh_noground_v4/AGB_tindex_master_mosaic.json'

## TODO: make these dicts
AGB_MOSAIC_JSON_FN_DICT = {
    'c2020_v1.0': '',
    '2019_v1.9' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2024_v5/AGB_H30_2019/2019_fullboreal_2019lidar/AGB_tindex_master_mosaic.json',
    '2020_v1.9' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2024_v5/AGB_H30_2020/atl08_v6_fullboreal_min5000_90p_local/AGB_tindex_master_mosaic.json',
    '2021_v1.9' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2024_v5/AGB_H30_2021/2021_fullboreal_2021lidar/AGB_tindex_master_mosaic.json',
    '2022_v1.9' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2024_v5/AGB_H30_2022/2022_fullboreal_2022lidar/AGB_tindex_master_mosaic.json',
    '2023_v1.9' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2024_v5/AGB_H30_2023/2023_full_2023lidar/AGB_tindex_master_mosaic.json',
    '2020_v2.0' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2024_v6/AGB_H30_2020/Version2_SD/AGB_tindex_master_mosaic.json',
    '2020_v2.1' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/dev_v1.5/AGB_H30_2020/full_run/AGB_tindex_master_mosaic.json',
    '2020_v2.2' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/dev_v1.5/AGB_S1H30_2020/full_run_no_uncert/AGB_tindex_master_mosaic.json',
}
AGB_TINDEX_FN_DICT = dict()
for key, value in AGB_MOSAIC_JSON_FN_DICT.items():
    if '_mosaic.json' in value:
        AGB_TINDEX_FN_DICT[key] = value.replace('_mosaic.json', '.csv')

######
###### Boreal Height mosaics
######
# c2020
#Ht_mosaic_json_fn = 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/AGB_tindex_master_height_mosaic.json'
#Ht_mosaic_json_fn = 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2023_v3/Ht_L30_2020/local_training_full/HT_tindex_master_mosaic.json'

## TODO: make these dicts
HT_MOSAIC_JSON_FN_DICT = {
    'c2020_v1.0': '',
    '2020_v2.0' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2024_v6/Ht_H30_2020/Version2_SD/HT_tindex_master_mosaic.json',
    '2020_v2.1_no_uncert' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/dev_v1.5/Ht_H30_2020/full_run_no_uncert/HT_tindex_master_mosaic.json',
    '2020_v2.1' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/dev_v1.5/Ht_H30_2020/full_run/HT_tindex_master_mosaic.json',
    '2020_v2.2' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/dev_v1.5/Ht_S1H30_2020/full_run_no_uncert/HT_tindex_master_mosaic.json'
}
HT_TINDEX_FN_DICT = dict()
for key, value in HT_MOSAIC_JSON_FN_DICT.items():
    if '_mosaic.json' in value:
        HT_TINDEX_FN_DICT[key] = value.replace('_mosaic.json', '.csv')

######
###### Harmonized Landsat-Sentinel L30 mosaics (Landsat only)
######
#############################
# HLS tindex from Laura's AGB DPS: nathanmthomas/DPS_tile_lists/HLS/c2020/HLS_stack_2022_v2/HLS_tindex_master.csv
# Final HLS comp for c2020  used for Boreal AGB c2020 <---UPDATE THIS TO INCLUDE NEW DATELINE TILES FROM L30
MS_COMP_VERSION = 'HLS_stack_2022_v2' 
HLS_tindex_c2020_master_fn = f's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/c2020/{MS_COMP_VERSION}/HLS_tindex_master.csv'
HLS_mosaic_c2020_json_fn   = f's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/c2020/{MS_COMP_VERSION}/HLS_tindex_master_mosaic.json'
# Dateline tiles (18) were run to append to this set (feb 2024)
HLS_mosaic_c2020datelines_json_fn = 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_L30_c2020/HLS_tindex_master_mosaic.json'

# Copy the dateline tiles
#!aws s3 cp s3://maap-ops-workspace/montesano/dps_output/do_HLS_stack_3-1-2/HLS_stack_2023_v1/HLS_L30_c2020/ s3://maap-ops-workspace/nathanmthomas/dps_output/do_HLS_stack_3-1-2_ubuntu/HLS_stack_2022_v2/2023/
#############################
###This doesnt exist######### they were 'ON-YEAR' 2020 - which have been re-done in montesano dps_output
# # Final HLS comp for c2020  
# MS_COMP_VERSION = 'HLS_stack_2023_v1'
# HLS_mosaic_c2020_json_fn = f's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/c2020/{MS_COMP_VERSION}/HLS_tindex_master_mosaic.json'
#############################
HLS_mosaic_c2020oldv1_json_fn   = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/spring2022/HLS_tindex_master_mosaic.json'

# initial HLS comp for c2020 built in fall 2022
HLS_mosaic_c2020oldv2_json_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/fall2022/HLS_stack_2022_v2/HLS_tindex_master_mosaic.json'
HLS_mosaic_c2020oldv2fix_json_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/fall2022_fix_irregulars/HLS_stack_2022_v2/12/HLS_tindex_master_mosaic.json'

# Fixed additional irregulars for c2020
HLS_mosaic_c2020_fixed_add_irregs_mosaic_json_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/c2020_fix_additional_irregulars/HLS_stack_2022_v2/HLS_tindex_master_mosaic.json'

# Test of HLS comp for late winter c2020
HLS_mosaic_c2020latewinter_json_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/c2020latewinter/HLS_stack_2022_v2/HLS_tindex_master_mosaic.json'

######
###### Harmonized Landsat-Sentinel H30 mosaics (Landsat L30 + Sentinel2 S30)
######
# Final HLS comp for c2015
HLS_mosaic_c2015_json_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/c2015/HLS_stack_2022_v2/11/HLS_tindex_master_mosaic.json' 

HLS_MOSAIC_JSON_FN_DICT = {
    'c2020oldv1':HLS_mosaic_c2020oldv1_json_fn,
    # This has a bunch a large artifacts
    'c2020oldv2':HLS_mosaic_c2020oldv2_json_fn,
    'c2020v2022nmt': HLS_mosaic_c2020_json_fn,
    # Copied from c2020v2022nmt so can be updated with datelines
    'c2020v2022pmm': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/c2020/HLS_stack_2022_v2/HLS_tindex_master_mosaic.json',
    'c2020v2022datelines': HLS_mosaic_c2020datelines_json_fn,
    # Good update
    # Updated with dateline : sum of c2020v2022pmm +  c2020v2022datelines
    'c2020v2022updated' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/c2020/HLS_stack_2022_v2/HLS_tindex_master_mosaic_updated.json',
    # Bad update - this is contaiminated with 10 c2015 tiles at the tail end of the file
    # Updated with dateline : sum of c2020v2022nmt +  c2020v2022datelines
    'c2020v2022updated_bad': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2022_v2/HLS_tindex_master_mosaic.json',
    '2015':  HLS_mosaic_c2015_json_fn,
    '2016': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2016/HLS_tindex_master_mosaic.json',
    '2017': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2017/HLS_tindex_master_mosaic.json',
    '2018': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2018/HLS_tindex_master_mosaic.json',
    '2018_test': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2018/mc0_mn50_07-01_08-31_2018_2018/HLS_tindex_master_mosaic.json',
    '2019': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2019/HLS_tindex_master_mosaic.json',
    '2020': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2020/HLS_tindex_master_mosaic.json',
    '2021': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2021/HLS_tindex_master_mosaic.json',
    '2022': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2022/HLS_tindex_master_mosaic.json',
    '2023': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2023/HLS_tindex_master_mosaic.json',
    '2024': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2024/HLS_tindex_master_mosaic.json',
}

HLS_TINDEX_FN_DICT = dict()
for key, value in HLS_MOSAIC_JSON_FN_DICT.items():
    if '_mosaic.json' in value:
        HLS_TINDEX_FN_DICT[key] = value.replace('_mosaic.json', '.csv')
    if '_mosaic_updated.json' in value:
        HLS_TINDEX_FN_DICT[key] = value.replace('_mosaic_updated.json', '_updated.csv')

###
### This is a 1-off update of the c2020 HLS L30 mosaic to include dateline tiles
### we use this for the Phase 3 c2020 Boreal AGB map update
###
UPDATE_C2020_HLS = False 

if UPDATE_C2020_HLS:
    
    import sys
    ICESAT2_BOREAL_REPO_PATH = '/projects/code/icesat2_boreal'
    ICESAT2_BOREAL_LIB_PATH = ICESAT2_BOREAL_REPO_PATH + '/lib'
    sys.path.append(ICESAT2_BOREAL_LIB_PATH)
    import ExtractUtils
    import shutil
    import pandas as pd
    
    # MosaicJSON Update
    #
    # Nathan's version of the HLS L30 c2020 mosaic
    #mosaic_json_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/c2020/HLS_stack_2022_v2/HLS_tindex_master_mosaic.json'

    # My version of the HLS L30 c2020 mosaic - this will be copied
    mosaic_json_fn = HLS_MOSAIC_JSON_FN_DICT['c2020v2022pmm']

    # Copied file renamed and will recceive the update
    updated_mosaic_json_fn = os.path.splitext(HLS_MOSAIC_JSON_FN_DICT['c2020v2022pmm'])[0] + '_updated.json'
    shutil.copy(mosaic_json_fn.replace('s3://maap-ops-workspace/shared/montesano','/projects/my-public-bucket'), 
                updated_mosaic_json_fn.replace('s3://maap-ops-workspace/shared/montesano','/projects/my-public-bucket'))

    # mosiac json update
    ExtractUtils.update_mosaic_json(updated_mosaic_json_fn, HLS_mosaic_c2020datelines_json_fn)
    
    # Tindex CSV Update
    #
    # A new updated CSV will be written
    updated_tindex_csv_fn = os.path.splitext(HLS_TINDEX_FN_DICT['c2020v2022pmm'])[0] + '_updated.csv'
    # For the tindex CSV of 'c2020updated' we need to manually update (append) with the datelines (like we did for the mosaic json)
    tmp_tindex = pd.concat([pd.read_csv(HLS_TINDEX_FN_DICT['c2020v2022datelines']),
                            pd.read_csv(HLS_TINDEX_FN_DICT['c2020v2022pmm'])
                  ])
    tmp_tindex.to_csv(updated_tindex_csv_fn.replace('s3://maap-ops-workspace/shared/montesano','/projects/my-public-bucket'))
    print(f"Update complete: {updated_tindex_csv_fn}")
    
######
###### SAR S1 tri-seasonal yearly composites (Sentinel1)
######
# SAR subtiles from GEE
S1_subtile_mosaic_json_fn = 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/SAR/EXPORT_GEE_v6/SAR_S1_2019/S1_subtile_tindex_master_mosaic.json'

# SAR stacked with boreal tiles
SAR_MOSAIC_JSON_FN_DICT = {
    '2019': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/run_build_stack/build_stack_v2023_2/build_stack_S1/SAR_S1_2019/S1_tindex_master_mosaic.json',
    '2020': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/run_build_stack/build_stack_v2023_2/build_stack_S1/SAR_S1_2020/S1_tindex_master_mosaic.json',
}
SAR_TINDEX_FN_DICT = dict()
for key, value in SAR_MOSAIC_JSON_FN_DICT.items():
    SAR_TINDEX_FN_DICT[key] = value.replace('_mosaic.json', '.csv')
    
######
###### ESA Worldcover Land Cover map 2020
######
#LC_mosaic_json_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/09/LC_tindex_master_mosaic.json'
## TODO: make these dicts
LC_MOSAIC_JSON_FN_DICT = {
    # Final LC for c2020 in Phase 2 (ORNL DAAC)
    'c2020orig'    : 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/LC/LC_tindex_master_mosaic.json',
    # updated for Phase 3; these have dateline tiles
    'c2020updated' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/run_build_stack/build_stack_v2023/LC_ESA_WC_2020/LC_tindex_master_mosaic.json'
}
LC_TINDEX_FN_DICT = dict()
for key, value in LC_MOSAIC_JSON_FN_DICT.items():
    LC_TINDEX_FN_DICT[key] = value.replace('_mosaic.json', '.csv')
    
######
###### Copernicus GLO30 Topography
######
TOPO_MOSAIC_JSON_FN_DICT = {
    # Final Topo for c2020 in Phase 2 (ORNL DAAC); these do not have dateline tiles; some are not be exactly 3000x3000 (chk near southern Ontario?)
    'c2020orig'    : 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/Topo_tindex_master_mosaic.json',
    # Updated for Phase 3; these have dateline tiles; all tiles are 3000x3000 BUT this causes some notdata at borders of some tiles?? (chk border edges of some northern Siberia tiles?)
    'c2020updated' :  's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/run_build_stack_topo/build_stack_v2023_2/CopernicusGLO30/Topo_tindex_master_mosaic.json',
    'c2020updated_v2':'s3://maap-ops-workspace/shared/montesano/DPS_tile_lists/run_build_stack_topo/build_stack_v2024_2/CopernicusGLO30/Topo_tindex_master_mosaic.json',
}
TOPO_TINDEX_FN_DICT = dict()
for key, value in TOPO_MOSAIC_JSON_FN_DICT.items():
    TOPO_TINDEX_FN_DICT[key] = value.replace('_mosaic.json', '.csv')

MISC_MOSAIC_JSON_FN_DICT = {
    'AGE_TP_2020':        's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/AGE/build_stack_v2023_2/AGE_TP_2020/AGE_tindex_master_mosaic.json',
    'TCCTREND_TP_2020':   's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/TCCTREND/build_stack_v2023_2/TCCTREND_TP_2020/TCCTREND_tindex_master_mosaic.json',
    'TCC_TP_2020':        's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/TCC/build_stack_v2023_2/TCC_TP_2020/TCC_tindex_master_mosaic.json',
    'FORESTAGE100m_2020': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/FORESTAGE100m/tile_forestage_v1/forestage_2020/FORESTAGE100m_tindex_master_mosaic.json',
    'FORESTAGE_BES_2020': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/FORESTAGE/build_stack_v2023_2/FORESTAGE_BES_2020/FORESTAGE_tindex_master_mosaic.json',
}

######
###### Dictionaries for building tindex files - these are helpful for providing the relevant info for re-running tindex and mosaic jsons
######
DICT_BUILD_TINDEX_ATL08_FILT = {
  'SET': 'ATL08',
 'USER': 'montesano',
 'ALG_NAME': 'process_atl08_boreal', 
 'ALG_VERSION': 'process_atl08_boreal',
 'VAR': 'ATL08_filt',
 'BATCH_NAME': '030m/2020',
 'YEAR_LIST': '2024',
 'DPS_MONTH_LIST': '02',
 'DPS_DAY_MIN': 1,
 'TILES_INDEX_PATH': '/projects/shared-buckets/montesano/databank/boreal_tiles_v004_model_ready.gpkg'
}
DICT_BUILD_TINDEX_ATL08_FILT_EXTRACT = {
  'SET': 'ATL08',
 'USER': 'montesano',
 'ALG_NAME': 'run_extract_atl08_covars', 
 'ALG_VERSION': 'extract_atl08_covars',
 'VAR': 'ATL08_filt_extract',
 'BATCH_NAME': '2020',
 'YEAR_LIST': '2024',
 'DPS_MONTH_LIST': '02 03',
 'DPS_DAY_MIN': 1,
 'TILES_INDEX_PATH': '/projects/shared-buckets/montesano/databank/boreal_tiles_v004_model_ready.gpkg'
}
DICT_BUILD_TINDEX_AGB = {
    'SET' : 'BOREAL_MAP',
    'USER' : 'lduncanson',
    'ALG_NAME' : 'run_boreal_biomass_map',
    'ALG_VERSION' : 'boreal_agb_2024_v6', 
    'VAR' : 'AGB',
    'BATCH_NAME' : 'AGB_H30_2020/Version2_SD',
    'YEAR_LIST': '2024',
    'DPS_MONTH_LIST' : '07 08 09 10',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
DICT_BUILD_TINDEX_HT = {
    'SET' : 'BOREAL_MAP',
    'USER' : 'lduncanson',
    'ALG_NAME' : 'run_boreal_biomass_map',
    'ALG_VERSION' : 'boreal_agb_2024_v6', 
    'VAR' : 'HT',
    'BATCH_NAME' : 'Ht_H30_2020/Version2_SD',
    'YEAR_LIST': '2024',
    'DPS_MONTH_LIST' : '07 08 09 10',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
# DICT_BUILD_TINDEX_HLS_L30_c2020_datelines = {
#     'SET' : 'HLS',
#     'USER' : 'montesano',
#     'ALG_NAME' : 'do_HLS_stack_3-1-2',
#     'ALG_VERSION' : 'HLS_stack_2023_v1',
#     'VAR' : 'HLS',
#     # In my bucket, this is ALWAYS used to identify output
#     'BATCH_NAME' : f'HLS_L30_c2020',
#     'YEAR_LIST': '2024',
#     'DPS_MONTH_LIST' : '02',        
#     'DPS_DAY_MIN' : 1 ,
#     'TILES_INDEX_PATH': boreal_tile_index_path
# }
# DICT_BUILD_TINDEX_AGB = {
#     'SET' : 'BOREAL_MAP',
#     'USER' : 'lduncanson',
#     'ALG_NAME' : 'run_boreal_biomass_map',
#     'ALG_VERSION' : 'boreal_agb_2023_v3',
#     'VAR' : 'AGB',
#     # In my bucket, this is ALWAYS used to identify output
#     'BATCH_NAME' : f'AGB_L30_2020/local_training_full',
#     'DPS_MONTH_LIST' : '12',        
#     'DPS_DAY_MIN' : 1 ,
#     'TILES_INDEX_PATH': boreal_tile_index_path
# }
# DICT_BUILD_TINDEX_AGB = {
#     'SET' : 'BOREAL_MAP',
#     'USER' : 'lduncanson',
#     'ALG_NAME' : 'run_boreal_biomass_map',
#     'ALG_VERSION' : 'boreal_agb_2024_v1',
#     'VAR' : 'AGB',
#     # In my bucket, this is ALWAYS used to identify output
#     'BATCH_NAME' : f'AGB_L30_2020/filter_swap_test',
#     'DPS_MONTH_LIST' : '02',        
#     'DPS_DAY_MIN' : 1 ,
#     'TILES_INDEX_PATH': boreal_tile_index_path
# }
# DICT_BUILD_TINDEX_AGB = {
#     'SET' : 'BOREAL_MAP',
#     'USER' : 'lduncanson',
#     'ALG_NAME' : 'run_boreal_biomass_map',
#     'ALG_VERSION' : 'boreal_agb_2024_v2',
#     'VAR' : 'AGB',
#     # In my bucket, this is ALWAYS used to identify output
#     'BATCH_NAME' : f'AGB_L30_2020/add_maxn_fullboreal',
#     'YEAR_LIST': '2024',
#     'DPS_MONTH_LIST' : '02',        
#     'DPS_DAY_MIN' : 1 ,
#     'TILES_INDEX_PATH': boreal_tile_index_path
# }
DICT_BUILD_TINDEX_SAR = {
    'SET' : 'SAR',
    'USER' : 'montesano',
    'ALG_NAME' : 'do_gee_download_by_subtile',
    'ALG_VERSION' : 'EXPORT_GEE_v6',
    'VAR' : 'S1_subtile',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : f'SAR_S1_2020',
    'YEAR_LIST': '2024',
    'DPS_MONTH_LIST' : '01',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': '/projects/my-public-bucket/DPS_tile_lists/SAR/EXPORT_GEE_v6/SAR_S1_2019/S1_gee_subtiles.gpkg'
}
DICT_BUILD_TINDEX_TCC2020 = {
    'SET' : 'TCC',
    'USER' : 'montesano',
    'ALG_NAME' : 'run_build_stack',
    'ALG_VERSION' : 'build_stack_v2023_2',
    'VAR' : 'TCC',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : 'TCC_TP_2020', # f'TCC_TP_{MAPYEAR}'
    'YEAR_LIST': '2024 2025',
    'DPS_MONTH_LIST' : '01 02 03 04 05 06 07 08 09 10 11 12', #'02',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
DICT_BUILD_TINDEX_TCC1984 = {
    'SET' : 'TCC',
    'USER' : 'montesano',
    'ALG_NAME' : 'run_build_stack',
    'ALG_VERSION' : 'build_stack_v2023_2',
    'VAR' : 'TCC',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : 'TCC_TP_1984',
    'YEAR_LIST': '2024 2025',
    'DPS_MONTH_LIST' : '01 02 03 04 05 06 07 08 09 10 11 12',         
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
DICT_BUILD_TINDEX_TCCTREND2020 = {
    'SET' : 'TCCTREND',
    'USER' : 'montesano',
    'ALG_NAME' : 'run_build_stack',
    'ALG_VERSION' : 'build_stack_v2023_2',
    'VAR' : 'TCCTREND',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : f'TCCTREND_TP_2020',
    'YEAR_LIST': '2024 2025',
    'DPS_MONTH_LIST' : '01 02 03 04 05 06 07 08 09 10 11 12', #'02',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
DICT_BUILD_TINDEX_AGE2020 = {
    'SET' : 'AGE',
    'USER' : 'montesano',
    'ALG_NAME' : 'run_build_stack',
    'ALG_VERSION' : 'build_stack_v2023_2',
    'VAR' : 'AGE',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : f'AGE_TP_2020',
    'YEAR_LIST': '2024 2025',
    'DPS_MONTH_LIST' : '01 02 03 04 05 06 07 08 09 10 11 12',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
DICT_BUILD_TINDEX_GLO30 = {
    'SET' : 'TOPO',
    'USER' : 'montesano',
    'ALG_NAME' : 'run_build_stack_topo',
    'ALG_VERSION' : 'build_stack_v2024_2',
    'VAR' : 'Topo',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : f'CopernicusGLO30',
    'YEAR_LIST': '2024',
    'DPS_MONTH_LIST' : '07',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
DICT_BUILD_TINDEX_FORESTAGE100m = {
    'SET' : 'FORESTAGE100m',
    'USER' : 'montesano', 
    'ALG_NAME' : 'run_tile_forestage',
    'ALG_VERSION' : 'tile_forestage_v1', 
    'VAR' : 'FORESTAGE100m',
    'BATCH_NAME' : 'forestage_2020',
    'YEAR_LIST': '2025',
    'DPS_MONTH_LIST' : '01',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
DICT_BUILD_TINDEX_FORESTAGE2020 = {
    'SET' : 'FORESTAGE',
    'USER' : 'montesano',
    'ALG_NAME' : 'run_build_stack',
    'ALG_VERSION' : 'build_stack_v2023_2',
    'VAR' : 'FORESTAGE',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : f'FORESTAGE_BES_2020',
    'YEAR_LIST': '2024 2025',
    'DPS_MONTH_LIST' : '01 02 03 04 05 06 07 08 09 10 11 12',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
###################################
## Functions
def print_tindex_vars(local_vars):
    return [print(i) for i in local_vars if 'tindex' in i][0]
def print_mosaic_vars(local_vars):
    return [print(i) for i in local_vars if 'mosaic_json_fn' in i or 'MOSAIC_JSON' in i][0]