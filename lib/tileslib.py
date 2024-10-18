import maplib_folium
import mosaiclib

#####
##### XYZ tiles mosaics from project results
#####
S1_2020summer_TILES_LAYER_DICT = maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['S1']['2020 HV summer'], 
                                            mosaiclib.SAR_MOSAIC_JSON_FN_DICT['2020'], 
                                            "HV from S1 summer composite: 2020", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,0.1", "bidx":"4", "colormap_name": "cubehelix"}
                                           )
S1_2019summer_TILES_LAYER_DICT = maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['S1']['2019 HV summer'], 
                                            mosaiclib.SAR_MOSAIC_JSON_FN_DICT['2019'], 
                                            "HV from S1 summer composite: 2019", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,0.1", "bidx":"4", "colormap_name": "cubehelix"}
                                           )
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

HLS_TILES_LAYER_DICT = {
                        '2016': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2016'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2016'],
                                                    "HLS 2016 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        '2020': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2020'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2020'],
                                                    "HLS 2020 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "BrBG"})
}

AGB_2020_TILE_LAYER_DICT = maplib_folium.make_tiles_layer_dict(
                                                       mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v2.0'], 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v2.0'], 
                                                       "AGB [Mg/ha] 2020", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"})

HT_2020_TILE_LAYER_DICT = maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.0'], 
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.0'], 
                                                    "Height [m] 2020", 
                                                    SHOW_CBAR=True, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   )