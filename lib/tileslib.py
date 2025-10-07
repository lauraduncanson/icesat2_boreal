import maplib_folium
import mosaiclib

'''
    Tile Layer Dictionaries
    -----------------------
        + these create folium 'tiles' layers of various project map results from XYZ tiles mosaics.
        + each dict holds a dict with all the info needed to map a layer:
          a. the TiTiler mosaic registration id (generated on the fly)
          b. the mosaic json s3 path
          c. the map legend string for the layer
          d. boolean for showing a colorbar on map for the layer
          e. parameters used to display map of the layer

    Using Tiles layers to provide API endpoints:
          To Use Tiles in QGIS, Geopandas.explore(), etc:
          tileslib.HT_TILE_LAYER_DICT['2020_v2.1']['layer'].tiles
'''

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
                                            "Stand Age in 2020 (Landsat record; TerraPulse)", 
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
                                            "Forest Age [enemble mean, 30m] 2020 (adapted from Besnard et al.)", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"1,250", "bidx":"1", "colormap_name": 'nipy_spectral'}
                                            ),
                    'DECPRED2015': maplib_folium.make_tiles_layer_dict(
                                            mosaiclib.TITILER_MOSAIC_REG_DICT['DECPRED']['2015'], 
                                            mosaiclib.MISC_MOSAIC_JSON_FN_DICT['DECPRED_AB_2015'], 
                                            "Deciduous fraction [%] 2015 (Massey et al.)", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,100", "bidx":"1", "colormap_name": 'cividis'}
                                            ),
                    # 'CACC_2020_v3.0_n05': maplib_folium.make_tiles_layer_dict(
                    #                         None, 
                    #                         mosaiclib.MISC_MOSAIC_JSON_FN_DICT['CACC_2020_v3.0_n05'], 
                    #                         "Carbon accumulation 2020 [MgC/ha/yr]", 
                    #                         SHOW_CBAR=True, 
                    #                         PARAMS_DICT = {"rescale": f"0,0.6", "bidx":"1", "colormap_name": 'turbo'}
                    #                         ),
                    'CACC_2020_v3.0_nsims050': maplib_folium.make_tiles_layer_dict(
                                            None, 
                                            mosaiclib.MISC_MOSAIC_JSON_FN_DICT['CACC_2020_v3.0_nsims050'], 
                                            "Carbon accumulation (v2) 2020 [MgC/ha/yr]", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,0.6", "bidx":"1", "colormap_name": 'turbo'}
                                            ),
                    'CACC_2020_v3.1_multiyr_nsims050': maplib_folium.make_tiles_layer_dict(
                                            None, 
                                            mosaiclib.MISC_MOSAIC_JSON_FN_DICT['CACC_2020_v3.1_multiyr_nsims050'], 
                                            "Carbon accumulation (v3) 2020 [MgC/ha/yr]", 
                                            SHOW_CBAR=True, 
                                            PARAMS_DICT = {"rescale": f"0,0.6", "bidx":"1", "colormap_name": 'turbo'}
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
                        # '2016': maplib_folium.make_tiles_layer_dict(
                        #                             mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2016'],
                        #                             mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2016'],
                        #                             "HLS 2016 NDVI", 
                        #                             SHOW_CBAR=True,
                        #                             PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        # '2019': maplib_folium.make_tiles_layer_dict(
                        #                             mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2019'],
                        #                             mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2019'],
                        #                             "HLS 2019 NDVI", 
                        #                             SHOW_CBAR=True,
                        #                             PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        # '2020': maplib_folium.make_tiles_layer_dict(
                        #                             mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2020'],
                        #                             mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2020'],
                        #                             "HLS 2020 NDVI", 
                        #                             SHOW_CBAR=True,
                        #                             PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        # '2021': maplib_folium.make_tiles_layer_dict(
                        #                             mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2021'],
                        #                             mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2021'],
                        #                             "HLS 2021 NDVI", 
                        #                             SHOW_CBAR=True,
                        #                             PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        # '2022': maplib_folium.make_tiles_layer_dict(
                        #                             mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2022'],
                        #                             mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2022'],
                        #                             "HLS 2022 NDVI", 
                        #                             SHOW_CBAR=True,
                        #                             PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        # '2023': maplib_folium.make_tiles_layer_dict(
                        #                             mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2023'],
                        #                             mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2023'],
                        #                             "HLS 2023 NDVI", 
                        #                             SHOW_CBAR=True,
                        #                             PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        # '2024': maplib_folium.make_tiles_layer_dict(
                        #                             mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2024'],
                        #                             mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024'],
                        #                             "HLS 2024 NDVI", 
                        #                             SHOW_CBAR=True,
                        #                             PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
                        # '2024CONUS': maplib_folium.make_tiles_layer_dict(
                        #                             mosaiclib.TITILER_MOSAIC_REG_DICT['HLS NDVI']['2024CONUS'],
                        #                             mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024CONUS'],
                        #                             "HLS 2024 NDVI", 
                        #                             SHOW_CBAR=True,
                        #                             PARAMS_DICT = {"rescale": f"0.3, 0.8", "bidx":"7", "colormap_name": "hsv"}),
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
                        '2019max': maplib_folium.make_tiles_layer_dict(
                                                    None,
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2019max'],
                                                    "HLS 2019max NBR2", 
                                                    SHOW_CBAR=False,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
                        '2024max': maplib_folium.make_tiles_layer_dict(
                                                    None,
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024max'],
                                                    "HLS 2024max NBR2", 
                                                    SHOW_CBAR=False,
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
                        '2025': maplib_folium.make_tiles_layer_dict(
                                                    None,
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2025'],
                                                    "HLS 2025 NBR2", 
                                                    SHOW_CBAR=False,
                                                    PARAMS_DICT = {"rescale": f"0.25, 0.45", "bidx":"13", "colormap_name": "nipy_spectral"}),
}
HLS_CNT_TILE_LAYER_DICT = {
                        '2019': maplib_folium.make_tiles_layer_dict(
                                                   None,
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2019'],
                                                    "HLS 2019 COUNT", 
                                                    SHOW_CBAR=False,
                                                    PARAMS_DICT = {"rescale": f"0, 15", "bidx":"22", "colormap_name": "magma"}),
                        '2019max': maplib_folium.make_tiles_layer_dict(
                                                    None,
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2019max'],
                                                    "HLS 2019max COUNT", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0, 15", "bidx":"22", "colormap_name": "magma"}),
                        '2024': maplib_folium.make_tiles_layer_dict(
                                                    None,
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024'],
                                                    "HLS 2024 COUNT", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0, 15", "bidx":"22", "colormap_name": "magma"}),
                        '2024max': maplib_folium.make_tiles_layer_dict(
                                                    None,
                                                    mosaiclib.HLS_MOSAIC_JSON_FN_DICT['2024max'],
                                                    "HLS 2024max COUNT", 
                                                    SHOW_CBAR=True,
                                                    PARAMS_DICT = {"rescale": f"0, 15", "bidx":"22", "colormap_name": "magma"}),
}
AGB_TILE_LAYER_DICT = {
                            # '2020_v2.0': maplib_folium.make_tiles_layer_dict(
                            #                            mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v2.0'], 
                            #                            mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v2.0'], 
                            #                            "AGBD [Mg/ha] (HLS+Topo) v2.0", 
                            #                            SHOW_CBAR=False, 
                            #                            PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            # '2020_v2.1': maplib_folium.make_tiles_layer_dict(
                            #                            mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v2.1'], 
                            #                            mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v2.1'], 
                            #                            "AGBD 2020 [Mg/ha] (HLS+Topo) v2.1", 
                            #                            SHOW_CBAR=False, 
                            #                            PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            # '2020_v2.2': maplib_folium.make_tiles_layer_dict(
                            #                            mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v2.2'], 
                            #                            mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v2.2'], 
                            #                            "AGBD [Mg/ha] (S1+HLS+Topo) v2.2", 
                            #                            SHOW_CBAR=False, 
                            #                            PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            # '2024_v2.1': maplib_folium.make_tiles_layer_dict(
                            #                            mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2024_v2.1'], 
                            #                            mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2024_v2.1'], 
                            #                            "AGBD 2024 [Mg/ha] (HLS+Topo) v2.1", 
                            #                            SHOW_CBAR=False, 
                            #                            PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2020_v3.0': maplib_folium.make_tiles_layer_dict(
                                                       mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v3.0'], 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v3.0'], 
                                                       "AGBD 2020 [Mg/ha] (HLS+Topo) v3.0", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2020_v3.0 std': maplib_folium.make_tiles_layer_dict(
                                                       mosaiclib.TITILER_MOSAIC_REG_DICT['AGB']['2020_v3.0'], 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v3.0'], 
                                                       "st. dev AGBD 2020 [Mg/ha] (HLS+Topo) v3.0", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 40", "bidx":"2", "colormap_name": "plasma"}),
                            '2020_v3.1': maplib_folium.make_tiles_layer_dict(
                                                       None,  
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v3.1'], 
                                                       "AGBD 2020 [Mg/ha] (HLS+Topo) v3.1", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2016_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2016_v3.1_multiyr'], 
                                                       "AGBD 2016 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2017_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2017_v3.1_multiyr'], 
                                                       "AGBD 2017 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2018_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2018_v3.1_multiyr'], 
                                                       "AGBD 2018 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2019_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2019_v3.1_multiyr'], 
                                                       "AGBD 2019 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2020_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None,  
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2020_v3.1_multiyr'], 
                                                       "AGBD 2020 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2021_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None,  
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2021_v3.1_multiyr'], 
                                                       "AGBD 2021 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2022_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2022_v3.1_multiyr'], 
                                                       "AGBD 2022 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2023_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2023_v3.1_multiyr'], 
                                                       "AGBD 2023 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2024_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2024_v3.1_multiyr'], 
                                                       "AGBD 2024 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
                            '2025_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.AGB_MOSAIC_JSON_FN_DICT['2025_v3.1_multiyr'], 
                                                       "AGBD 2025 [Mg/ha] (HLS+Topo) v3.1_multiyr", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 125", "bidx":"1", "colormap_name": "viridis"}),
}

HT_TILE_LAYER_DICT = {
                            # '2020_v2.0': maplib_folium.make_tiles_layer_dict(
                            #                         mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.0'], 
                            #                         mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.0'], 
                            #                         "Height [m] 2020 (HLS+Topo) v2.0", 
                            #                         SHOW_CBAR=False, 
                            #                         # Success only with standard colormap_name
                            #                         PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                            #                        ),
                            # '2020_v2.1_no_uncert': maplib_folium.make_tiles_layer_dict(
                            #                         mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.1_no_uncert'], 
                            #                         mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.1_no_uncert'], 
                            #                         "Height [m] 2020 (HLS+Topo) v2.1 (no uncerts)", 
                            #                         SHOW_CBAR=False, 
                            #                         # Success only with standard colormap_name
                            #                         PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                            #                        ),
                            # '2020_v2.1': maplib_folium.make_tiles_layer_dict(
                            #                         mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.1'], 
                            #                         mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.1'], 
                            #                         "Height [m] 2020 (HLS+Topo) v2.1", 
                            #                         SHOW_CBAR=False, 
                            #                         # Success only with standard colormap_name
                            #                         PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                            #                        ),
                            # '2020_v2.2': maplib_folium.make_tiles_layer_dict(
                            #                         mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v2.2'], 
                            #                         mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v2.2'], 
                            #                         "Height [m] 2020 (S1+HLS+Topo) v2.2", 
                            #                         SHOW_CBAR=False, 
                            #                         # Success only with standard colormap_name
                            #                         PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                            #                        ),
                            '2019_v3.0': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2019_v3.0'],
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2019_v3.0'], 
                                                    "Height [m] 2019 (HLS+Topo) v3.o", 
                                                    SHOW_CBAR=False, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2020_v3.0': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_v3.0'],
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v3.0'], 
                                                    "Height [m] 2020 (HLS+Topo) v3.o", 
                                                    SHOW_CBAR=True, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2021_v3.0': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2021_v3.0'],
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2021_v3.0'], 
                                                    "Height [m] 2021 (HLS+Topo) v3.o", 
                                                    SHOW_CBAR=False, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2022_v3.0': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2022_v3.0'],
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2022_v3.0'], 
                                                    "Height [m] 2022 (HLS+Topo) v3.o", 
                                                    SHOW_CBAR=False, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2023_v3.0': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2023_v3.0'],
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2023_v3.0'], 
                                                    "Height [m] 2023 (HLS+Topo) v3.o", 
                                                    SHOW_CBAR=False, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2024_v3.0': maplib_folium.make_tiles_layer_dict(
                                                    mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2024_v3.0'],
                                                    mosaiclib.HT_MOSAIC_JSON_FN_DICT['2024_v3.0'], 
                                                    "Height [m] 2024 (HLS+Topo) v3.o", 
                                                    SHOW_CBAR=False, 
                                                    # Success only with standard colormap_name
                                                    PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
                                                   ),
                            '2020_v3.1': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_v3.1'], 
                                                       "Height [m] 2020 [m] (HLS+Topo) v3.1", 
                                                       SHOW_CBAR=True, 
                                                       PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}),
                            '2024_v3.1_multiyr': maplib_folium.make_tiles_layer_dict(
                                                       None, 
                                                       mosaiclib.HT_MOSAIC_JSON_FN_DICT['2024_v3.1_multiyr'], 
                                                       "Height [m] 2024 [m] (HLS+Topo) v3.1", 
                                                       SHOW_CBAR=False, 
                                                       PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}),
    # # ------------------------ TESTS ---------------------------------
    #                         '2024_quebec_agu24': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2024_quebec_agu24'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2024_quebec_agu24'], 
    #                                                 "Height [m] 2024 (HLS+Topo) Quebec for AGU24", 
    #                                                 SHOW_CBAR=True, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 50", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_neon38': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_neon38'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_neon38'], 
    #                                                 "Height [m] 2020 (HLS+Topo) NEON 38", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 50", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_neon38_sd': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_neon38'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_neon38'], 
    #                                                 "Height [m] St. Dev 2020 (HLS+Topo) NEON 38", 
    #                                                 SHOW_CBAR=True, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 5", "bidx":"2", "colormap_name": "plasma"}
    #                                                ),
    #                         '2020_neon38atl08': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_neon38atl08'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_neon38atl08'], 
    #                                                 "Height [m] 2020 (HLS+Topo) NEON 38 from ICESat-2 ATL08", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 50", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_neon38l2a': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_neon38l2a'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_neon38l2a'], 
    #                                                 "Height [m] 2020 (HLS+Topo) NEON 38 from GEDI L2A", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 50", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_niter_250_ntree_50': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_niter_250_ntree_50'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_niter_250_ntree_50'], 
    #                                                 "Height [m] 2020 (HLS+Topo) niter=250, ntree=50", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_niter_250_ntree_100': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_niter_250_ntree_100'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_niter_250_ntree_100'], 
    #                                                 "Height [m] 2020 (HLS+Topo) niter=250, ntree=100", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_niter_250_ntree_100_no_uncert_moss_lichen_0': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_niter_250_ntree_100_no_uncert_moss_lichen_0'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_niter_250_ntree_100_no_uncert_moss_lichen_0'], 
    #                                                 "Height [m] 2020 (HLS+Topo) niter=250, ntree=100, no_uncert_moss_lichen_0", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_remove_short_veg': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_remove_short_veg'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_remove_short_veg'], 
    #                                                 "Height [m] 2020 (HLS+Topo) remove_short_veg", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_zero_short_veg_height': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_zero_short_veg_height'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_zero_short_veg_height'], 
    #                                                 "Height [m] 2020 (HLS+Topo) zero_short_veg_height (slope>25)", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_zero_short_veg_height_slope15': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_zero_short_veg_height_slope15'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_zero_short_veg_height_slope15'], 
    #                                                 "Height [m] 2020 (HLS+Topo) zero_short_veg_height (slope>15)", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_niter_250_ntrees_50_10_cores': maplib_folium.make_tiles_layer_dict(
    #                                                 mosaiclib.TITILER_MOSAIC_REG_DICT['HT']['2020_niter_250_ntrees_50_10_cores'], 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_niter_250_ntrees_50_10_cores'], 
    #                                                 "Height [m] 2020 (HLS+Topo) niter_250_ntrees_50_10_cores", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2019_multiyearmodel': maplib_folium.make_tiles_layer_dict(
    #                                                 None, 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2019_multiyearmodel'], 
    #                                                 "Height [m] 2019 (HLS+Topo) multiyearmodel", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
    #                         '2020_multiyearmodel': maplib_folium.make_tiles_layer_dict(
    #                                                 None, 
    #                                                 mosaiclib.HT_MOSAIC_JSON_FN_DICT['2020_multiyearmodel'], 
    #                                                 "Height [m] 2020 (HLS+Topo) multiyearmodel", 
    #                                                 SHOW_CBAR=False, 
    #                                                 # Success only with standard colormap_name
    #                                                 PARAMS_DICT = {"rescale": f"0, 30", "bidx":"1", "colormap_name": "inferno"}
    #                                                ),
}