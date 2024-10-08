import maplib_folium
import mosaiclib

#####
##### XYZ tiles mosaics from project results
#####

LC_TILES_LAYER_DICT = maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['LC']['c2020updated'], 
                                            None, 
                                            "ESA Worldcover 2020 v1.0", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"10,100", "bidx":"1", "colormap_name": "tab20"}
                                           )
TOPO_TILES_LAYER_DICT = maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['TOPO']['c2020updated_v2'], 
                                              None, 
                                              "Copernicus GLODEM", 
                                              SHOW_CBAR=False, 
                                              PARAMS_DICT = {"rescale": f"0,1", "bidx":"3", "colormap_name": 'gist_gray'}
                                             )
#TOPO_TILES_LAYER_DICT['layer'].options.update({'opacity': 0.25})

MAX_AGB = 125
CMAP_AGB = 'viridis'
VERSION_NAME = 'Version2_SD'
DPS_IDENTIFIER_ =     f'boreal_agb_2024_v6/AGB_H30_2020/{VERSION_NAME}'
AGB_2020_MOSAIC_JSON_FN = f's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/{DPS_IDENTIFIER_}/AGB_tindex_master_mosaic.json'
AGB_2020_TILE_LAYER_DICT = maplib_folium.make_tiles_layer_dict(
                                                       mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v2.0'], 
                                                       None, #AGB_2020_MOSAIC_JSON_FN, 
                                                       "AGB [Mg/ha] 2020", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0,{MAX_AGB}", "bidx":"1", "colormap_name": f"{CMAP_AGB}"})

MAX_HT = 30
CMAP_HT = 'inferno'
VERSION_NAME = 'Version2_SD'
DPS_IDENTIFIER_HT =     f'boreal_agb_2024_v6/Ht_H30_2020/{VERSION_NAME}'
HT_2020_MOSAIC_JSON_FN = f's3://maap-ops-workspace/shared/montesano/DPS_tile_lists/BOREAL_MAP/{DPS_IDENTIFIER_HT}/HT_tindex_master_mosaic.json'
HEIGHT_2020_TILE_LAYER_DICT = maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.0'], 
                                                    None, #HT_2020_MOSAIC_JSON_FN, 
                                                    "Height [m] 2020", SHOW_CBAR=True, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0,{MAX_HT}", "bidx":"1", "colormap_name": f"{CMAP_HT}"}
                                                   )