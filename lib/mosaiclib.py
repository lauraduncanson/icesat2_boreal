######
######
###### Library of all mosaic json files associated with various iterations of the ABoVE Boreal AGB Density maps
######
###### variables hold the paths to the mosaic json files on MAAP
###### import this into other notebooks: from mosaiclib import *

boreal_tile_index_path = '/projects/shared-buckets/montesano/databank/boreal_tiles_v004.gpkg'
MINI_DATELINE_TILES  = [400400,391600,382300,372800,363400,354000,4199500,4180700,4161900]
LARGE_DATELINE_TILES = [3540,3634,3728,3823,3916,4004,41995,41807,41619]

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
###### Boreal filtered ATL08 geodataframes (stored as CSVs)
######
#ATL08_filt_tindex_master_fn = 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/ATL08_filt_tindex_master.csv'
#ATL08_filt_tindex_master_fn = 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/with_atl03_rh/ATL08_filt_tindex_master.csv'
#ATL08_filt_tindex_json_fn= "/projects/my-public-bucket/DPS_tile_lists/ATL08_filt_tindex_master.json"
ATL08_FILT_TINDEX_FN_DICT = {
    # Final ATL08_filt
    'c2020spring2022' : 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/ATL08_filt/c2020/tile_atl08/ATL08_filt_tindex_master.csv', # note: in nathan's now
    'c2020fall2022v1' : 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/with_gedi_rh/ATL08_filt_tindex_master.csv',
    'c2020fall2022v2' : 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/with_atl03_rh/ATL08_filt_tindex_master.csv'
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
AGB_MOSAIC_JSON_FN_DICT = dict()
AGB_TINDEX_FN_DICT = dict()

######
###### Boreal Height mosaics
######
# c2020
#Ht_mosaic_json_fn = 's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/fall2022/AGB_tindex_master_height_mosaic.json'
Ht_mosaic_json_fn = 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/boreal_agb_2023_v3/Ht_L30_2020/local_training_full/HT_tindex_master_mosaic.json'

######
###### Harmonized Landsat-Sentinel L30 mosaics (Landsat only)
######
#############################
# Final HLS comp for c2020  <---UPDATE THIS TO INCLUDE NEW DATELINE TILES FROM L30
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
    'c2020oldv2':HLS_mosaic_c2020oldv2_json_fn,
    'c2020v2022': HLS_mosaic_c2020_json_fn,
    'c2020v2022datelines': HLS_mosaic_c2020datelines_json_fn,
    # Updated with dateline : sum of nathan's c2020v2022 +  c2020v2022datelines
    'c2020updated': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2022_v2/HLS_tindex_master_mosaic.json',
    '2015':  HLS_mosaic_c2015_json_fn,
    '2016': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2016/HLS_tindex_master_mosaic.json',
    '2017': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2017/HLS_tindex_master_mosaic.json',
    '2018': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2018/HLS_tindex_master_mosaic.json',
    '2019': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2019/HLS_tindex_master_mosaic.json',
    '2020': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2020/HLS_tindex_master_mosaic.json',
    '2021': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2021/HLS_tindex_master_mosaic.json',
    '2022': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2022/HLS_tindex_master_mosaic.json',
    '2023': 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_H30_2023/HLS_tindex_master_mosaic.json'
} # DO THIS: s3://maap-ops-workspace/shared/montesano/DPS_tile_lists/HLS/HLS_stack_2023_v1/HLS_L30_c2020/HLS_tindex_master_mosaic.json <--same as what we used but with a few new dateline tiles
HLS_TINDEX_FN_DICT = dict()
for key, value in HLS_MOSAIC_JSON_FN_DICT.items():
    HLS_TINDEX_FN_DICT[key] = value.replace('_mosaic.json', '.csv')

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
    'c2020updated' : 's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/run_build_stack_topo/build_stack_v2023_2/CopernicusGLO30/Topo_tindex_master_mosaic.json'
}
TOPO_TINDEX_FN_DICT = dict()
for key, value in TOPO_MOSAIC_JSON_FN_DICT.items():
    TOPO_TINDEX_FN_DICT[key] = value.replace('_mosaic.json', '.csv')

######
###### Build tindex dictionaries - these are helpful for providing the relevant info for re-running tindex and mosaic jsons
######
DICT_BUILD_TINDEX_HLS_L30_c2020_datelines = {
    'SET' : 'HLS',
    'USER' : 'montesano',
    'ALG_NAME' : 'do_HLS_stack_3-1-2',
    'ALG_VERSION' : 'HLS_stack_2023_v1',
    'VAR' : 'HLS',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : f'HLS_L30_c2020',
    'YEAR': 2024,
    'DPS_MONTH_LIST' : '02',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
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
DICT_BUILD_TINDEX_AGB = {
    'SET' : 'BOREAL_MAP',
    'USER' : 'lduncanson',
    'ALG_NAME' : 'run_boreal_biomass_map',
    'ALG_VERSION' : 'boreal_agb_2024_v2',
    'VAR' : 'AGB',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : f'AGB_L30_2020/add_maxn_fullboreal',
    'YEAR': 2024,
    'DPS_MONTH_LIST' : '02',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}
DICT_BUILD_TINDEX_SAR = {
    'SET' : 'SAR',
    'USER' : 'montesano',
    'ALG_NAME' : 'do_gee_download_by_subtile',
    'ALG_VERSION' : 'EXPORT_GEE_v6',
    'VAR' : 'S1_subtile',
    # In my bucket, this is ALWAYS used to identify output
    'BATCH_NAME' : f'SAR_S1_2020',
    'YEAR': 2024,
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
    'YEAR': 2024,
    'DPS_MONTH_LIST' : '02',        
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
    'YEAR': 2024,
    'DPS_MONTH_LIST' : '02',        
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
    'YEAR': 2024,
    'DPS_MONTH_LIST' : '02',        
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
    'YEAR': 2024,
    'DPS_MONTH_LIST' : '02',        
    'DPS_DAY_MIN' : 1 ,
    'TILES_INDEX_PATH': boreal_tile_index_path
}

###################################
## Functions
def print_tindex_vars(local_vars):
    return [print(i) for i in local_vars if 'tindex' in i][0]
def print_mosaic_vars(local_vars):
    return [print(i) for i in local_vars if 'mosaic_json_fn' in i or 'MOSAIC_JSON' in i][0]