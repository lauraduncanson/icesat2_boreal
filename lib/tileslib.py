import maplib_folium
import mosaiclib

#####
##### XYZ tiles mosaics from project results
#####

MISC_TILES_LAYER_DICT = {
                    'S1_2020summer': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['S1']['2020 HV summer'], 
                                            mosaiclib.SAR_MOSAIC_JSON_FN_DICT['2020'], 
                                            "HV from S1 summer composite: 2020", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,0.1", "bidx":"4", "colormap_name": "cubehelix"}
                                           ),
                    'S1_2019summer': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['S1']['2019 HV summer'], 
                                            mosaiclib.SAR_MOSAIC_JSON_FN_DICT['2019'], 
                                            "HV from S1 summer composite: 2019", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,0.1", "bidx":"4", "colormap_name": "cubehelix"}
                                           ),
                    'LC2020': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['LC']['c2020updated'], 
                                            None, 
                                            "ESA Worldcover 2020 v1.0", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"10,100", "bidx":"1", "colormap_name": "tab20"}
                                           ),
                    'LC2020CONUS': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['LC']['2020CONUS'], 
                                            mosaiclib.LC_MOSAIC_JSON_FN_DICT['2020CONUS'],  
                                            "ESA Worldcover 2020 v1.0", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"10,100", "bidx":"1", "colormap_name": "tab20"}
                                           ),
                    'TOPO2020': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['TOPO']['c2020updated_v2'], 
                                            None, 
                                            "Copernicus GLODEM", 
                                            SHOW_CBAR=False, 
                                            PARAMS_DICT = {"rescale": f"0,1", "bidx":"3", "colormap_name": 'gist_gray'}
                    ),
                    'TOPO2020CONUS': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['TOPO']['2020CONUS'], 
                                            mosaiclib.TOPO_MOSAIC_JSON_FN_DICT['2020CONUS'], 
                                            "Copernicus GLODEM", 
                                            SHOW_CBAR=False, 
                                            PARAMS_DICT = {"rescale": f"0,1", "bidx":"3", "colormap_name": 'gist_gray'}
                    ),
                    #TOPO_TILES_LAYER_DICT['layer'].options.update({'opacity': 0.25})

                    'AGE2020': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['AGE']['2020'], 
                                            mosaiclib.MISC_MOSAIC_JSON_FN_DICT['AGE_TP_2020'], 
                                            "Stand Age in 2020 (Landsat Record)", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,250", "bidx":"1", "colormap_name": 'nipy_spectral'}
                                            ),
                    'TCCTREND2020': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['TCCTREND']['2020'], 
                                            mosaiclib.MISC_MOSAIC_JSON_FN_DICT['TCCTREND_TP_2020'], 
                                            "Tree canopy cover trend 1984-2020", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"-2,2", "bidx":"1", "colormap_name": 'BrBG'}
                                            ),
                    'TCC2020': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['TCC']['2020'], 
                                            mosaiclib.MISC_MOSAIC_JSON_FN_DICT['TCC_TP_2020'], 
                                            "Tree canopy cover 2020", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,75", "bidx":"1", "colormap_name": 'YlGn'}
                                            ),
                    'FORESTAGE100m2020': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['FORESTAGE100m']['2020'], 
                                            mosaiclib.MISC_MOSAIC_JSON_FN_DICT['FORESTAGE100m_2020'], 
                                            "FORESTAGE [enemble mean, 1 ha.] 2020 (Besnard et al.)", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"1,250", "bidx":"1", "colormap_name": 'nipy_spectral'}
                                            ),
                    'FORESTAGE2020': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['FORESTAGE']['2020'], 
                                            mosaiclib.MISC_MOSAIC_JSON_FN_DICT['FORESTAGE_BES_2020'], 
                                            "FORESTAGE [enemble mean, 30m] 2020 (Besnard et al.)", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"1,250", "bidx":"1", "colormap_name": 'nipy_spectral'}
                                            ),
}
#     if mosaic_json_dict['tp_tcc2020slope_json_s3_fn'] is not None:
#         TCC2020SLOPE_MAX = 2
#         TCC2020SLOPE_MIN = -2
#         TCC2020SLOPE_COLORBAR = 'BrBG'
#         cmap = matplotlib.cm.get_cmap(TCC2020SLOPE_COLORBAR, 12)
#         colormap_TCC2020SLOPE = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(TCC2020SLOPE_MIN, TCC2020SLOPE_MAX)
#         colormap_TCC2020SLOPE.caption = "Tree Canopy Cover trend (1984-2020)"
#         m1.add_child(colormap_TCC2020SLOPE)
HLS_NDVI_TILES_LAYER_DICT = {
                        '2016': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2016'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2016'],
                                                    "HLS 2016 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        '2019': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2019'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2019'],
                                                    "HLS 2019 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        '2020': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2020'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2020'],
                                                    "HLS 2020 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        '2021': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2021'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2021'],
                                                    "HLS 2021 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        '2022': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2022'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2022'],
                                                    "HLS 2022 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        '2023': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2023'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2023'],
                                                    "HLS 2023 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        '2024': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2024'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024'],
                                                    "HLS 2024 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        '2024CONUS': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2024CONUS'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024CONUS'],
                                                    "HLS 2024 NDVI", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
}
HLS_NBR2_TILES_LAYER_DICT = {
                        '2016': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2016'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2016'],
                                                    "HLS 2016 NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                        '2018_test': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2018_test'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2018_test'],
                                                    "HLS 2018_test NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                        '2019_orig': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2019'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2019_orig'],
                                                    "HLS 2019 orig NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                        '2019': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2019'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2019'],
                                                    "HLS 2019 NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                       '2020': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2020'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2020'],
                                                    "HLS 2020 NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                       '2021': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2021'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2021'],
                                                    "HLS 2021 NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                       '2022': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2022'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2022'],
                                                    "HLS 2022 NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                       '2023': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2023'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2023'],
                                                    "HLS 2023 NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                        '2024': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2024'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024'],
                                                    "HLS 2024 NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                        '2024CONUS': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NBR2']['2024CONUS'],
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024CONUS'],
                                                    "HLS 2024 NBR2", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
}

AGB_TILE_LAYER_DICT = {
                            '2020_v2.0': maplib_folium.make_tiles_layer_dict(
                                                       mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v2.0'], 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v2.0'], 
                                                       "AGB [Mg/ha] (HLS+Topo) v2.0", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2020_v2.1': maplib_folium.make_tiles_layer_dict(
                                                       mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v2.1'], 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v2.1'], 
                                                       "AGB 2020 [Mg/ha] (HLS+Topo) v2.1", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2020_v2.2': maplib_folium.make_tiles_layer_dict(
                                                       mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v2.2'], 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v2.2'], 
                                                       "AGB [Mg/ha] (S1+HLS+Topo) v2.2", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2024_v2.1': maplib_folium.make_tiles_layer_dict(
                                                       mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2024_v2.1'], 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2024_v2.1'], 
                                                       "AGB 2024 [Mg/ha] (HLS+Topo) v2.1", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
}

HT_TILE_LAYER_DICT = {
                            '2020_v2.0': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.0'], 
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.0'], 
                                                    "Height [m] 2020 (HLS+Topo) v2.0", 
                                                    SHOW_CBAR=True, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2020_v2.1_no_uncert': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.1_no_uncert'], 
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.1_no_uncert'], 
                                                    "Height [m] 2020 (HLS+Topo) v2.1 (no uncerts)", 
                                                    SHOW_CBAR=False, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2020_v2.1': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.1'], 
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.1'], 
                                                    "Height [m] 2020 (HLS+Topo) v2.1", 
                                                    SHOW_CBAR=True, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2020_v2.2': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.2'], 
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.2'], 
                                                    "Height [m] 2020 (S1+HLS+Topo) v2.2", 
                                                    SHOW_CBAR=True, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   )
}