import geopandas
import pandas as pd
import os

import branca
import branca.colormap as cm
import matplotlib.cm

from folium import Map, TileLayer, GeoJson, LayerControl, Icon, Marker, features, Figure, CircleMarker
from folium import plugins

# Get a basemap
tiler_basemap_googleterrain = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'
tiler_basemap_gray =          'http://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
tiler_basemap_image =         'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
tiler_basemap_natgeo =        'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
basemaps = {
   'Google Terrain' : TileLayer(
    tiles = tiler_basemap_googleterrain,
    attr = 'Google',
    name = 'Google Terrain',
    overlay = False,
    control = True
   ),
    'basemap_gray' : TileLayer(
        tiles=tiler_basemap_gray,
        opacity=1,
        name="ESRI gray",
        attr="MAAP",
        overlay=False
    ),
    'Imagery' : TileLayer(
        tiles=tiler_basemap_image,
        opacity=1,
        name="ESRI imagery",
        attr="MAAP",
        overlay=False
    ),
    'ESRINatGeo' : TileLayer(
    tiles=tiler_basemap_natgeo,
    opacity=1,
    name='ESRI Nat. Geo.',
    attr='ESRI',
    overlay=False
    )
}

def local_to_s3(url, user = 'nathanmthomas', type='public'):
    ''' A Function to convert local paths to s3 urls'''
    if type == 'public':
        replacement_str = f's3://maap-ops-workspace/shared/{user}'
    else:
        replacement_str = f's3://maap-ops-workspace/{user}'
    return url.replace(f'/projects/my-{type}-bucket', replacement_str)

def GET_BOREAL_TILE_LAYER(boreal_tile_index, tiles_remove, boreal_tiles_style={'fillColor': '#e41a1c', 'color': '#e41a1c', 'weight' : 0.5, 'opacity': 1, 'fillOpacity': 0}):
    boreal_tile_index_layer = GeoJson(
            data=boreal_tile_index[~boreal_tile_index.tile_num.isin(tiles_remove)].to_crs("EPSG:4326").to_json(),
            style_function=lambda x:boreal_tiles_style,
            name="Boreal tiles",
            tooltip=features.GeoJsonTooltip(
                fields=['tile_num'],
                aliases=['Tile num:'],
            )
        )
    return boreal_tile_index_layer


def MAP_DPS_RESULTS(tiler_mosaic, boreal_tile_index, 
                    tile_index_matches,  
                    tile_index_check, 
                    MATCH_TILES_NAME='Match tiles', 
                    CHECK_TILES_NAME='Check tiles', 
                    plots = None,
                    mosaic_json_dict = {
                                        'agb_mosaic_json_s3_fn':    's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/AGB_tindex_master_mosaic.json',
                                        'topo_mosaic_json_s3_fn':   's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/Topo_tindex_master_mosaic.json',
                                        'mscomp_mosaic_json_s3_fn': 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS_tindex_master_mosaic.json',
                                        'worldcover_json_s3_fn': None,
                                        'tp_standage2020_json_s3_fn': None,
                                        'tp_tcc2020_json_s3_fn': None,
                                        'tp_tcc2020slope_json_s3_fn': None,
                                        'tp_tcc2020pvalue_json_s3_fn': None
                                    },
                    mscomp_rgb_dict = None,
                    #ecoboreal_geojson = '/projects/shared-buckets/nathanmthomas/Ecoregions2017_boreal_m.geojson',
                    ecoboreal_geojson = '/projects/shared-buckets/nathanmthomas/analyze_agb/input_zones/wwf_circumboreal_Dissolve.geojson',
                    max_AGB_display = 50, max_AGBSE_display = 20,
                    MS_BAND_DICT = {
                        'name': 'NDVI',
                        'num': 8,
                        'min': 0,
                        'max': 1,
                        'cmap': 'viridis',
                        'legend_name': 'NDVI composite'
                    },
                    tiles_remove = [41995, 41807, 41619], # geo abyss,
                    SHOW_WIDGETS=False,
                    TOPO_OPACITY=0.15,
                    map_width=1000, map_height=500,
                    ADD_TILELAYER = None,
                    HLS_TILELAYER_LIST = None
                   ):
    
    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None:
        
        # TODO: find other valid 'colormap_names' for the tiler url that also work with cm.linear.xxxx.scale()
        agb_colormap = 'viridis'#'RdYlGn_11' #'RdYlGn' #'nipy_spectral'
        agb_tiles = f"{tiler_mosaic}?url={mosaic_json_dict['agb_mosaic_json_s3_fn']}&rescale=0,{max_AGB_display}&bidx=1&colormap_name={agb_colormap}"
        #colormap_AGB = cm.linear.viridis.scale(0, max_AGB_display).to_step(25)
        cmap = matplotlib.cm.get_cmap(agb_colormap, 25)
        colormap_AGB = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, max_AGB_display)
        colormap_AGB.caption = 'Mean of Aboveground Biomass Density [Mg/ha]'
        
        agb_se_colormap = 'plasma'
        agb_se_tiles = f"{tiler_mosaic}?url={mosaic_json_dict['agb_mosaic_json_s3_fn']}&rescale=0,{max_AGBSE_display}&bidx=2&colormap_name={agb_se_colormap}"
        #colormap_AGBSE = cm.linear.plasma.scale(0, 20).to_step(5)
        cmap = matplotlib.cm.get_cmap(agb_se_colormap, 25)
        colormap_AGBSE = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, max_AGBSE_display)
        colormap_AGBSE.caption = 'Standard Error of Aboveground Biomass Density [Mg/ha]'
        
    if ADD_TILELAYER is not None:
        if isinstance(ADD_TILELAYER, list):
            ADD_TILELAYER_LIST = ADD_TILELAYER
        else:
            ADD_TILELAYER_LIST = [ADD_TILELAYER]
            
        colormap_ADDED_TILELAYER_list = []
        for ADD_TILELAYER in ADD_TILELAYER_LIST:
            #cmap = matplotlib.cm.get_cmap('plasma', 30)
            #colormap_Ht = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, 30)
            #colormap_Ht.caption = 'Vegetation Height (m)'
            cmap = matplotlib.cm.get_cmap(ADD_TILELAYER["cmap"], 25)
            colormap_ADDED_TILELAYER = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, ADD_TILELAYER["max_val"])
            colormap_ADDED_TILELAYER.caption = ADD_TILELAYER["caption"]
            colormap_ADDED_TILELAYER_list.append(colormap_ADDED_TILELAYER)

    # Get Vector layers
    #boreal_geojson = '/projects/shared-buckets/lduncanson/misc_files/wwf_circumboreal_Dissolve.geojson'#'/projects/shared-buckets/nathanmthomas/boreal.geojson' 
    #boreal_geojson = '/projects/shared-buckets/lduncanson/misc_files/Ecoregions2017_boreal_m.geojson'
    #boreal = geopandas.read_file(boreal_geojson)

    # Style Vector Layers
    ecoboreal_style = {'fillColor': 'gray', 'color': 'gray'}
    boreal_style = {'fillColor': 'gray', 'color': 'gray'}
    boreal_subset_style = {'fillColor': 'red', 'color': 'red'}

    if True:

        ecoboreal = geopandas.read_file(ecoboreal_geojson)
        # Reproject Vector Layers
        p1, p2, clat, clon = [50, 70, 40, 160]
        proj_str_aea = '+proj=aea +lat_1={:.2f} +lat_2={:.2f} +lat_0={:.2f} +lon_0={:.2f}'.format(p1, p2, clat, clon)
        ecoboreal_aea = ecoboreal.to_crs(proj_str_aea)
        # Apply a buffer
        ecoboreal_aea_buf = ecoboreal_aea["geometry"].buffer(1e5)
        # Go back to GCS
        ecoboreal_buf = ecoboreal_aea_buf.to_crs(boreal_tile_index.crs)
        ecoboreal_layer = GeoJson(ecoboreal, name="Boreal extent from Ecoregions", style_function=lambda x:ecoboreal_style)
        #GeoJson(ecoboreal_aea_buf, name="Boreal extent from Ecoregions", style_function=lambda x:ecoboreal_style).add_to(m1)
        #GeoJson(boreal, name="Boreal extent", style_function=lambda x:boreal_style).add_to(m1)

    # Map the Layers
    #Map_Figure=Figure(width=map_width,height=map_height)
    Map_Figure=Figure()
    #------------------
    m1 = Map(
        width=map_width,height=map_height,
        #tiles="Stamen Toner",
        tiles='',
        location=(60, 5),
        zoom_start=3, 
        control_scale = True
    )
    Map_Figure.add_child(m1)

    boreal_tiles_style = {'fillColor': '#e41a1c', 'color': '#e41a1c', 'weight' : 0.5, 'opacity': 1, 'fillOpacity': 0}
    dps_subset_style = {'fillColor': '#377eb8', 'color': '#377eb8', 'weight' : 0.75, 'opacity': 1, 'fillOpacity': 0.5}
    dps_check_style = {'fillColor': 'red', 'color': 'red'}
    
    # Set colormaps for legends
    # Choose colormap names from this set: 
    # plt.cm.datad.keys()
        #     dict_keys(['Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges',\
        #                          'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', \
        #                          'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', \
        #                          'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'jet', \
        #                          'nipy_spectral', 'ocean', 'pink', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'terrain', 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', \
        #                          'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'])
    
    if mosaic_json_dict['tp_tcc2020pvalue_json_s3_fn'] is not None:
        TCC2020PVALUE_MAX = 1
        TCC2020PVALUE_COLORBAR = 'hot'
        cmap = matplotlib.cm.get_cmap(TCC2020PVALUE_COLORBAR, 12)
        colormap_TCC2020PVALUE = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, TCC2020PVALUE_MAX)
        colormap_TCC2020PVALUE.caption = "Tree Canopy Cover trend p-value"
        m1.add_child(colormap_TCC2020PVALUE)
        
    if mosaic_json_dict['tp_tcc2020slope_json_s3_fn'] is not None:
        TCC2020SLOPE_MAX = 2
        TCC2020SLOPE_MIN = -2
        TCC2020SLOPE_COLORBAR = 'BrBG'
        cmap = matplotlib.cm.get_cmap(TCC2020SLOPE_COLORBAR, 12)
        colormap_TCC2020SLOPE = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(TCC2020SLOPE_MIN, TCC2020SLOPE_MAX)
        colormap_TCC2020SLOPE.caption = "Tree Canopy Cover trend (1984-2020)"
        m1.add_child(colormap_TCC2020SLOPE)
        
    if mosaic_json_dict['tp_standage2020_json_s3_fn'] is not None:
        STANDAGE2020_MAX = 35
        STANDAGE2020_COLORBAR = 'jet'
        cmap = matplotlib.cm.get_cmap(STANDAGE2020_COLORBAR, 12)
        colormap_STANDAGE2020 = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, STANDAGE2020_MAX)
        colormap_STANDAGE2020.caption = "Stand Age (yrs in 2020)"
        m1.add_child(colormap_STANDAGE2020)
        
    if mosaic_json_dict['tp_tcc2020_json_s3_fn'] is not None:
        TCC2020_MAX = 75
        TCC2020_COLORBAR = 'YlGn'
        cmap = matplotlib.cm.get_cmap(TCC2020_COLORBAR, 12)
        colormap_tcc2020 = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, TCC2020_MAX)
        colormap_tcc2020.caption = "Tree Canopy Cover (%, 2020)"
        m1.add_child(colormap_tcc2020)
    
    if mosaic_json_dict['worldcover_json_s3_fn'] is not None:
        cols_worldcover = ["black","#006400","#ffbb22","#ffff4c","#f096ff","#fa0000","#b4b4b4","#f0f0f0","#0064c8","#0096a0","#00cf75","#fae6a0"]
        names_worldcover = ['No Data','Trees', 'Shrubland', 'Grassland','Cropland','Built-up','Barren / sparse vegetation','Snow and ice','Open water','Herbaceous wetland','Mangroves','Moss and lichen']
        values_worldcover = [0,10,20,30,40,50,60,70,80,90,95,100]
        colormap_worldcover_dict = dict(zip([str(n) for n in values_worldcover], cols_worldcover))
        colormap_worldcover = cm.StepColormap(colors = cols_worldcover, vmin=min(values_worldcover), vmax=max(values_worldcover), index=values_worldcover, caption = 'ESA Worldcover v1')
        m1.add_child(colormap_worldcover)
        
    if mosaic_json_dict['mscomp_mosaic_json_s3_fn'] is not None:
        cmap = matplotlib.cm.get_cmap(MS_BAND_DICT["cmap"], 25)
        colormap_MSCOMP = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(MS_BAND_DICT["min"], MS_BAND_DICT["max"])
        colormap_MSCOMP.caption = MS_BAND_DICT["name"]
        m1.add_child(colormap_MSCOMP)
        
    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None:
        m1.add_child(colormap_AGB)
        m1.add_child(colormap_AGBSE)

    #GeoJson(atl08_gdf, name="ATL08"
    #       ).add_to(m)

    boreal_tile_index_layer = GET_BOREAL_TILE_LAYER(boreal_tile_index, tiles_remove, boreal_tiles_style)

    tile_matches_layer = GeoJson(
            data=tile_index_matches,
            style_function=lambda x:dps_subset_style,
            name=f"{MATCH_TILES_NAME} completed",
            tooltip=features.GeoJsonTooltip(
                fields=['tile_num'],
                aliases=['Tile num:'],
            )
        )

    if tile_index_check is not None and len(tile_index_check) > 0:
        tile_index_check_layer = GeoJson(
                data=tile_index_check,
                style_function=lambda x:dps_check_style,
                name=f"{CHECK_TILES_NAME} tiles"
            )

    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None:
        agb_tiles_layer = TileLayer(
            tiles=agb_tiles,
            opacity=1,
            name="AGB",
            attr="MAAP",
            overlay=True
        )
        agb_tiles_layer.add_to(m1)

        agb_se_tiles_layer = TileLayer(
            tiles=agb_se_tiles,
            opacity=1,
            name="AGB SE",
            attr="MAAP",
            overlay=True
        )
        agb_se_tiles_layer.add_to(m1)
        
    if mscomp_rgb_dict is not None:
        
        mscomp_tiles_layer_red = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['mscomp_mosaic_json_s3_fn']}&rescale=0.01,{mscomp_rgb_dict['red_bandmax']}&bidx={mscomp_rgb_dict['red_bandnum']}&colormap_name=reds",
            opacity=0.33,
            name=f"MS Composite: {mscomp_rgb_dict['red_bandnum']}",
            attr="MAAP",
            overlay=True
        )
        mscomp_tiles_layer_red.add_to(m1)
        mscomp_tiles_layer_green = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['mscomp_mosaic_json_s3_fn']}&rescale=0.01,{mscomp_rgb_dict['green_bandmax']}&bidx={mscomp_rgb_dict['green_bandnum']}&colormap_name=greens",
            opacity=0.33,
            name=f"MS Composite: {mscomp_rgb_dict['green_bandnum']}",
            attr="MAAP",
            overlay=True
        )
        mscomp_tiles_layer_green.add_to(m1)
        mscomp_tiles_layer_blue = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['mscomp_mosaic_json_s3_fn']}&rescale=0.01,{mscomp_rgb_dict['blue_bandmax']}&bidx={mscomp_rgb_dict['blue_bandnum']}&colormap_name=blues",
            opacity=0.33,
            name=f"MS Composite: {mscomp_rgb_dict['blue_bandnum']}",
            attr="MAAP",
            overlay=True
        )
        mscomp_tiles_layer_blue.add_to(m1)
        
    elif mosaic_json_dict['mscomp_mosaic_json_s3_fn'] is not None:
        mscomp_tiles_layer = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['mscomp_mosaic_json_s3_fn']}&rescale={MS_BAND_DICT['min']},{MS_BAND_DICT['max']}&bidx={MS_BAND_DICT['num']}&colormap_name={MS_BAND_DICT['cmap']}",
            opacity=1,
            name=MS_BAND_DICT['legend_name'],
            attr="MAAP",
            overlay=True
        )
        mscomp_tiles_layer.add_to(m1)
        
    ###########################    
    # TILE LAYERS
    if mosaic_json_dict['tp_tcc2020pvalue_json_s3_fn'] is not None:
        tcc2020pvalue_tiles_layer = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['tp_tcc2020pvalue_json_s3_fn']}&rescale=0,{TCC2020PVALUE_MAX}&bidx=1&colormap_name={TCC2020PVALUE_COLORBAR.lower()}", # <---- THIS IS WORKING, but DOESNT MATCH THE CUSTOM COLORBAR WE NEED
            opacity=1,
            name="Tree canopy cover trend significance",
            attr="TerraPulse",
            overlay=True
        )
        tcc2020pvalue_tiles_layer.add_to(m1)  
        
    if mosaic_json_dict['tp_tcc2020slope_json_s3_fn'] is not None:
        tcc2020slope_tiles_layer = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['tp_tcc2020slope_json_s3_fn']}&rescale={TCC2020SLOPE_MIN},{TCC2020SLOPE_MAX}&bidx=1&colormap_name={TCC2020SLOPE_COLORBAR.lower()}", # <---- THIS IS WORKING, but DOESNT MATCH THE CUSTOM COLORBAR WE NEED
            opacity=1,
            name="Tree canopy cover trend (1984-2020)",
            attr="TerraPulse",
            overlay=True
        )
        tcc2020slope_tiles_layer.add_to(m1)  
        
    if mosaic_json_dict['tp_standage2020_json_s3_fn'] is not None:
        standage2020_tiles_layer = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['tp_standage2020_json_s3_fn']}&rescale=0,{STANDAGE2020_MAX}&bidx=1&colormap_name={STANDAGE2020_COLORBAR}", # <---- THIS IS WORKING, but DOESNT MATCH THE CUSTOM ; try this: https://developmentseed.org/titiler/examples/code/tiler_with_custom_colormap/COLORBAR WE NEED
            opacity=1,
            name="Stand age 2020",
            attr="TerraPulse",
            overlay=True
        )
        standage2020_tiles_layer.add_to(m1)
        
    if mosaic_json_dict['tp_tcc2020_json_s3_fn'] is not None:
        tcc2020_tiles_layer = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['tp_tcc2020_json_s3_fn']}&rescale=0,{TCC2020_MAX}&bidx=1&colormap_name={TCC2020_COLORBAR.lower()}", # <---- THIS IS WORKING, but DOESNT MATCH THE CUSTOM COLORBAR WE NEED
            opacity=1,
            name="Tree canopy cover 2020",
            attr="TerraPulse",
            overlay=True
        )
        tcc2020_tiles_layer.add_to(m1)
        
    if mosaic_json_dict['worldcover_json_s3_fn'] is not None:
        # encode the colormap so it works with a mosaic json?
        import urllib
        import json
        # colormap_worldcover_dict = [
        #         ((0, 10), '#006400'),
        #         ((10, 20) , '#ffbb22'),
        #         ((20, 30) , '#ffff4c'),
        #         ((30, 40) , '#f096ff'),
        #         ((40, 50) , '#fa0000'),
        #         ((50, 60) , '#b4b4b4'),
        #         ((60, 70) , '#f0f0f0'),
        #         ((70, 80) , '#0064c8'),
        #         ((80, 90) , '#0096a0'),
        #         ((90, 95) , '#00cf75'),
        #         ((95, 100) , '#fae6a0'),
        #         ((100, 255), '#000000')
        #     ]
        colormap_worldcover_dict = dict(zip([str(n) for n in values_worldcover], cols_worldcover))
        colormap_encode = urllib.parse.urlencode({"colormap": json.dumps(colormap_worldcover_dict)})
        colormap_worldcover = cm.linear.Set1_09.scale(0, 100).to_step(len(values_worldcover))
        worldcover_tiles_layer = TileLayer(
            #TODO: try this: https://developmentseed.org/titiler/examples/code/tiler_with_custom_colormap/
            #tiles= f"{tiler_mosaic}?url={mosaic_json_dict['worldcover_json_s3_fn']}&rescale=10,100&bidx=1&{colormap_encode}",
            #TODO: get this to work?
            #tiles= f"{tiler_mosaic}?url={mosaic_json_dict['worldcover_json_s3_fn']}&rescale=10,100&bidx=1&colormap={colormap_worldcover_dict}", # <---- THIS IS NOT WORKING
            #TODO: try this: https://python-visualization.github.io/folium/latest/advanced_guide/colormaps.html
            
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['worldcover_json_s3_fn']}&rescale=10,100&bidx=1&colormap_name=tab20", # <---- THIS IS WORKING, but DOESNT MATCH THE CUSTOM COLORBAR WE NEED
            opacity=1,
            name="Worldcover",
            attr="ESA",
            overlay=True
        )
        worldcover_tiles_layer.add_to(m1)
    
    # Add the additional tile layer to map with its colorbar
    if HLS_TILELAYER_LIST is not None:
        for TILELAYER in HLS_TILELAYER_LIST:
            TILELAYER.add_to(m1)
        # Just need to add the colorbar once    
        m1.add_child(colormap_ADDED_TILELAYER)
    if ADD_TILELAYER is not None:
        for i, ADD_TILELAYER in enumerate(ADD_TILELAYER_LIST):
            ADD_TILELAYER["layer"].add_to(m1)
            print(f"Adding layer {ADD_TILELAYER['caption']}...")
            if ADD_TILELAYER["show_cbar"]:
                m1.add_child(colormap_ADDED_TILELAYER_list[i])
        
    # Overlay topo last with an opacity     
    if mosaic_json_dict['topo_mosaic_json_s3_fn'] is not None:
        topo_tiles_layer = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['topo_mosaic_json_s3_fn']}&rescale=0,1&bidx=3&colormap_name=gist_gray", #tsri
            #tiles= f"{tiler_mosaic}?url={mosaic_json_dict['topo_mosaic_json_s3_fn']}&rescale=0,2&bidx=5&colormap_name=tab10", #slopemask
            opacity=TOPO_OPACITY,
            name="Topo Solar Rad. Idx",
            attr="MAAP",
            overlay=True
        )
        topo_tiles_layer.add_to(m1)
        
    # Add custom basemaps
    basemaps['Google Terrain'].add_to(m1)
    basemaps['Imagery'].add_to(m1)
    basemaps['ESRINatGeo'].add_to(m1)
    basemaps['basemap_gray'].add_to(m1)

    ecoboreal_layer.add_to(m1)

    # Layers are added on top. Last layer is top layer
    boreal_tile_index_layer.add_to(m1)
    tile_matches_layer.add_to(m1)

    if tile_index_check is not None and len(tile_index_check) > 0:
        tile_index_check_layer.add_to(m1)

    #tile_matches_n_obs.add_to(m1)    
    
    # Add reference plots
    if plots is not None and len(plots) > 0:
        
        #pal_heightref_cmap = cm.LinearColormap(colors = ['black','#636363','#fc8d59','#fee08b','#ffffbf','#d9ef8b','#91cf60','#1a9850'], vmin=0, vmax=35)
        pal_heightref_cmap = cm.LinearColormap(colors=['blue','white','red'], vmin=-10, vmax=10)
        pal_heightref_cmap.caption = 'Forest height from field'
        for lat, lon, ref_ht, pred_ht, diff_ht, diff_yr in zip(plots.geometry.y, plots.geometry.x, plots.ref_ht, plots.value_ht_L30_2020, plots.diff_ht, plots.diff_yr):
            plot = CircleMarker(location=[lat, lon],
                                radius = 10,
                                weight=0.75,
                                tooltip=f"Ref ht: {str(round(ref_ht,2))}\nPred. ht: {str(round(pred_ht,2))}m\nDiff: {str(round(diff_ht,2))}m",
                                fill=True,
                                #fill_color=getfill(h_can),
                                color = pal_heightref_cmap(diff_ht),
                                opacity=1,
                                    overlay=True,
                                name="Plots"
                                
                   )
            plot.add_to(m1)

    if SHOW_WIDGETS:
        plugins.Geocoder().add_to(m1)
        
    LayerControl().add_to(m1)
    plugins.Geocoder(position='bottomright').add_to(m1)
    plugins.Fullscreen(position='bottomleft').add_to(m1)
    plugins.MousePosition().add_to(m1)
    
    if SHOW_WIDGETS:
        minimap = plugins.MiniMap()
        m1.add_child(minimap)
        #m1.add_child(colormap_AGBSE)

    return m1

def map_tile_n_obs(tindex_master_fn='s3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/ATL08_filt_tindex_master.csv', 
                   map_name = '# of filtered ATL08 obs.',
                   max_n_obs=15000, map_width=1000, map_height=200, 
                   boreal_tile_index_path = '/projects/shared-buckets/nathanmthomas/boreal_tiles_v003.gpkg'):
    
    import pandas as pd
    import geopandas
    import branca.colormap as cm
    from folium import Map, TileLayer, GeoJson, LayerControl, Icon, Marker, features, Figure, CircleMarker
    
    # Build up a dataframe from the list of dps output files
    tindex_master = pd.read_csv(tindex_master_fn)

    # Get all boreal tiles
    boreal_tile_index = geopandas.read_file(boreal_tile_index_path)
    #boreal_tile_index.astype({'layer':'int'})
    boreal_tile_index.rename(columns={"layer":"tile_num"}, inplace=True)
    boreal_tile_index["tile_num"] = boreal_tile_index["tile_num"].astype(int)

    bad_tiles = [3540,3634,3728,3823,3916,4004] #Dropping the tiles near antimeridian that reproject poorly.

    boreal_tile_index = boreal_tile_index[~boreal_tile_index['tile_num'].isin(bad_tiles)]
    tile_matches = boreal_tile_index.merge(tindex_master[~tindex_master['tile_num'].isin(bad_tiles)], how='right', on='tile_num')

    nobs_cmap = cm.LinearColormap(colors=cm.linear.RdYlGn_11.colors, vmin=0, vmax=max_n_obs)

    tile_matches['color'] = [nobs_cmap(n_obs) for n_obs in tile_matches.n_obs]

    Map_Figure3=Figure(width=map_width,height=map_height)
    
    m3 = Map(
        #tiles="Stamen Toner",
        tiles='',
        location=(60, 5),
        zoom_start=2
    )
    Map_Figure3.add_child(m3)

    tile_matches_n_obs = GeoJson(
        tile_matches,
        style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            #'color' : feature['properties']['color'],
            'color' : 'black',
            'weight' : 1,
            'fillOpacity' : 0.85, 
            },
        name="ATL08 filt tiles: n_obs",
        tooltip=features.GeoJsonTooltip(
                fields=['tile_num','n_obs','local_path'],
                aliases=['Tile:','# obs.:','path:'],
            )
        )
    
        # Add custom basemaps
    basemaps['basemap_gray'].add_to(m3)
    basemaps['Google Terrain'].add_to(m3)
    basemaps['Imagery'].add_to(m3)
    basemaps['ESRINatGeo'].add_to(m3)

    tile_matches_n_obs.add_to(m3)
    colormap_nobs= nobs_cmap.to_step(15)
    colormap_nobs.caption = map_name
    m3.add_child(colormap_nobs)

    LayerControl().add_to(m3)

    return m3

def map_tile_n_scenes(tindex_master_fn='s3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS/fall2022/HLS_stack_2022_v2/HLS_input_params.csv', 
                   map_name = '# of HLS scenes',
                    N_SCENES_FIELD_NAME = "num_scenes",
                   max_n_obs=125, map_width=1000, map_height=200, 
                   boreal_tile_index_path = '/projects/shared-buckets/nathanmthomas/boreal_tiles_v003.gpkg'):
    
    import pandas as pd
    import geopandas
    import branca.colormap as cm
    from folium import Map, TileLayer, GeoJson, LayerControl, Icon, Marker, features, Figure, CircleMarker
    
    # Build up a dataframe from the list of dps output files
    tindex_master = pd.read_csv(tindex_master_fn)

    # Get all boreal tiles
    boreal_tile_index = geopandas.read_file(boreal_tile_index_path)
    #boreal_tile_index.astype({'layer':'int'})
    boreal_tile_index.rename(columns={"layer":"tile_num"}, inplace=True)
    boreal_tile_index["tile_num"] = boreal_tile_index["tile_num"].astype(int)

    bad_tiles = [3540,3634,3728,3823,3916,4004] #Dropping the tiles near antimeridian that reproject poorly.

    boreal_tile_index = boreal_tile_index[~boreal_tile_index['tile_num'].isin(bad_tiles)]
    tile_matches = boreal_tile_index.merge(tindex_master[~tindex_master['tile_num'].isin(bad_tiles)], how='right', on='tile_num')

    nobs_cmap = cm.LinearColormap(colors=cm.linear.RdYlGn_11.colors, vmin=0, vmax=max_n_obs)

    tile_matches['color'] = [nobs_cmap(n_obs) for n_obs in tile_matches[N_SCENES_FIELD_NAME]]

    Map_Figure3=Figure(width=map_width,height=map_height)
    
    m3 = Map(
        #tiles="Stamen Toner",
        tiles='',
        location=(60, 5),
        zoom_start=2
    )
    Map_Figure3.add_child(m3)

    tile_matches_n_obs = GeoJson(
        tile_matches,
        style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            #'color' : feature['properties']['color'],
            'color' : 'black',
            'weight' : 1,
            'fillOpacity' : 0.5,
            },
        name="HLS # scenes",
        tooltip=features.GeoJsonTooltip(
                fields=['tile_num',N_SCENES_FIELD_NAME,'run_type'],
                aliases=['Tile:','# scenes.:','run:'],
            )
        )
    
        # Add custom basemaps
    basemaps['basemap_gray'].add_to(m3)
    basemaps['Google Terrain'].add_to(m3)
    basemaps['Imagery'].add_to(m3)
    basemaps['ESRINatGeo'].add_to(m3)

    tile_matches_n_obs.add_to(m3)
    colormap_nobs= nobs_cmap.to_step(15)
    colormap_nobs.caption = map_name
    m3.add_child(colormap_nobs)

    LayerControl().add_to(m3)

    return m3

def map_tile_atl08(TILE_OF_INTEREST_LIST, tiler_mosaic, boreal_tindex_master,
                  DPS_DATA_USER = 'lduncanson', ATL08_filt_tindex_master_fn = f'/projects/shared-buckets/lduncanson/DPS_tile_lists/ATL08_filt_tindex_master.csv', DO_NIGHT=True,
                  mosaic_json_dict = {
                                        'agb_mosaic_json_s3_fn':    's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/AGB_tindex_master_mosaic.json',
                                        'topo_mosaic_json_s3_fn':   's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/Topo_tindex_master_mosaic.json',
                                        'mscomp_mosaic_json_s3_fn': 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS_tindex_master_mosaic.json'
                                    },
                   map_width = 100, map_height=600, OVERVIEW_MAP= True,
                   max_AGB_display = 50, max_AGBSE_display = 20, MAX_HEIGHT = 10
                  ):
    
    import branca.colormap as cm
    pal_height_cmap = cm.LinearColormap(colors = ['black','#636363','#fc8d59','#fee08b','#ffffbf','#d9ef8b','#91cf60','#1a9850'], vmin=0, vmax=MAX_HEIGHT)
    pal_height_cmap.caption = 'Vegetation height from  ATL08 @ 30 m (h_can; rh98)'
    pal_height_cmap
    
    # Set colormaps
    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None and tiler_mosaic is not None:
        
        # TODO: find other valid 'colormap_names' for the tiler url that also work with cm.linear.xxxx.scale()
        agb_colormap = 'viridis'#'RdYlGn_11' #'RdYlGn' #'nipy_spectral'
        agb_tiles = f"{tiler_mosaic}?url={mosaic_json_dict['agb_mosaic_json_s3_fn']}&rescale=0,{max_AGB_display}&bidx=1&colormap_name={agb_colormap}"
        #colormap_AGB = cm.linear.viridis.scale(0, max_AGB_display).to_step(25)
        cmap = matplotlib.cm.get_cmap(agb_colormap, 25)
        colormap_AGB = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, max_AGB_display)
        colormap_AGB.caption = 'Mean of Aboveground Biomass Density [Mg/ha]'
        
        agb_se_colormap = 'plasma'
        agb_se_tiles = f"{tiler_mosaic}?url={mosaic_json_dict['agb_mosaic_json_s3_fn']}&rescale=0,{max_AGBSE_display}&bidx=2&colormap_name={agb_se_colormap}"
        #colormap_AGBSE = cm.linear.plasma.scale(0, 20).to_step(5)
        cmap = matplotlib.cm.get_cmap(agb_se_colormap, 25)
        colormap_AGBSE = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(0, max_AGBSE_display)
        colormap_AGBSE.caption = 'Standard Error of Aboveground Biomass Density [Mg/ha]'
    
    #DPS_DATA_TYPE = 'ATL08_filt' #"Topo" "Landsat" "ATL08" "AGB"
     
    #AGB_tindex_master_fn = f's3://maap-ops-workspace/shared/{DPS_DATA_USER}/DPS_tile_lists/AGB_tindex_master.csv'
    
    if isinstance(ATL08_filt_tindex_master_fn, pd.DataFrame):
        print('Input is dataframe')
        atl08_gdf = ATL08_filt_tindex_master_fn.to_crs(4326)
        if 'h_canopy' in atl08_gdf.columns: atl08_gdf['h_can'] = atl08_gdf['h_canopy']
        atl08_gdf['lon'] = atl08_gdf.geometry.x
        atl08_gdf['lat'] = atl08_gdf.geometry.y
    else:
        print(ATL08_filt_tindex_master_fn)

        # Build up a dataframe from the list of dps output files
        #AGB_tindex_master = pd.read_csv(AGB_tindex_master_fn)
        #AGB_tindex_master['s3'] = [local_to_s3(local_path, user=DPS_DATA_USER, type = 'private') for local_path in AGB_tindex_master['local_path']]

        ATL08_filt_tindex_master = pd.read_csv(ATL08_filt_tindex_master_fn)
        if 's3_path' in ATL08_filt_tindex_master.columns:
            ATL08_filt_tindex_master['s3'] = ATL08_filt_tindex_master['s3_path']
        else:
            ATL08_filt_tindex_master['s3'] = [local_to_s3(local_path, user=DPS_DATA_USER, type = 'private') for local_path in ATL08_filt_tindex_master['local_path']]

        if TILE_OF_INTEREST_LIST[0] not in ATL08_filt_tindex_master.tile_num.to_list():
            print(f'Tile {TILE_OF_INTEREST_LIST[0]} has not yet been added to this list.')
            return None

        # Get the CSV fn for tile
        ATL08_filt_csv_fn = ATL08_filt_tindex_master['s3'].loc[ATL08_filt_tindex_master.tile_num.isin(TILE_OF_INTEREST_LIST)].tolist()[0]
        print(ATL08_filt_csv_fn)

        # Get corresponding ATL08 filtered csv
        atl08_df = pd.read_csv(ATL08_filt_csv_fn)
        atl08_gdf = geopandas.GeoDataFrame(atl08_df, crs="EPSG:4326", geometry = geopandas.points_from_xy(atl08_df.lon, atl08_df.lat) )
        
        
        if DO_NIGHT:
            print(f'Percentage of night (night_flg=1) ATL08 obs: \t\t{round(len(atl08_gdf[atl08_gdf.night_flg == 1]) / len(atl08_gdf),3) *100}%')
        #print(f'Percentage of water (ValidMask=0) ATL08 obs: \t\t{round(len(atl08_gdf[atl08_gdf.ValidMask == 0]) / len(atl08_gdf),3) *100}%')
        #print(f'Percentage of water (slopemask=0) ATL08 obs: \t\t{round(len(atl08_gdf[atl08_gdf.slopemask == 0]) / len(atl08_gdf),3) *100}%')
    
    print(f'\nNum. of ATL08 obs. in tile {TILE_OF_INTEREST_LIST}: \t{len(atl08_gdf)}')
    print(round(atl08_gdf.lat.mean(),4), round(atl08_gdf.lon.mean(),4))

    # Map the Layers
    #Map_Figure=Figure(width=map_width,height=map_height)
    Map_Figure=Figure()
    #------------------
    boreal_tile_of_interest_gdf = boreal_tindex_master[boreal_tindex_master.tile_num.isin(TILE_OF_INTEREST_LIST)].to_crs(4326)
    m2 = Map(
        tiles='',
        #location=(atl08_gdf.lat.mean(), atl08_gdf.lon.mean()),
        location = (boreal_tile_of_interest_gdf.geometry.centroid.y.median(), boreal_tile_of_interest_gdf.geometry.centroid.x.median()),
        zoom_start=8,
        control_scale = True
    )
    Map_Figure.add_child(m2)
    
    # Add boreal tiles
    boreal_tiles_index_master_layer = GET_BOREAL_TILE_LAYER(boreal_tindex_master.to_crs(4326), [], {'fillColor': '#e41a1c', 'color': 'black', 'weight' : 0.25, 'opacity': 1, 'fillOpacity': 0})
    boreal_tiles_index_master_layer.add_to(m2)
    boreal_tile_index_layer = GET_BOREAL_TILE_LAYER(boreal_tile_of_interest_gdf, [], {'fillColor': '#e41a1c', 'color': '#e41a1c', 'weight' : 1, 'opacity': 1, 'fillOpacity': 0})
    boreal_tile_index_layer.add_to(m2)

    if DO_NIGHT:
        atl08_gdf = atl08_gdf[atl08_gdf.night_flg == 1]
    #for lat, lon, ValidMask, slopemask, h_can in zip(atl08_gdf.lat, atl08_gdf.lon, atl08_gdf.ValidMask, atl08_gdf.slopemask, atl08_gdf.h_can):
    for lat, lon, h_can in zip(atl08_gdf.lat, atl08_gdf.lon, atl08_gdf.h_can):
        ATL08_obs_night = CircleMarker(location=[lat, lon],
                                radius = 10,
                                weight=5,
                                tooltip=str(round(h_can,2))+" m",
                                fill=True,
                                #fill_color=getfill(h_can),
                                color = pal_height_cmap(h_can),
                                #color = getcolor(ValidMask),
                                opacity=1,
                                    overlay=True,
                                name="ATL08 night obs"
                                
                   )

        ATL08_obs_night.add_to(m2)

    # Add custom basemaps
    basemaps['basemap_gray'].add_to(m2)
    basemaps['Google Terrain'].add_to(m2)
    basemaps['Imagery'].add_to(m2)
    basemaps['ESRINatGeo'].add_to(m2)
    
    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None and tiler_mosaic is not None:
        agb_tiles_layer = TileLayer(
            tiles=agb_tiles,
            opacity=1,
            name="Boreal AGB",
            attr="MAAP",
            overlay=True
        )
        agb_tiles_layer.add_to(m2)
        agb_se_tiles_layer = TileLayer(
            tiles=agb_se_tiles,
            opacity=1,
            name="Boreal AGB SE",
            attr="MAAP",
            overlay=True
        )
        agb_se_tiles_layer.add_to(m2)

    # Layers are added underneath. Last layer is bottom layer


    #tile_matches_missing_layer.add_to(m2)
    #tile_matches_layer.add_to(m2)
    if OVERVIEW_MAP:
        minimap = plugins.MiniMap()
        m2.add_child(minimap)
        
    m2.add_child(pal_height_cmap)
    
    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None and tiler_mosaic is not None:
        m2.add_child(colormap_AGB)
        m2.add_child(colormap_AGBSE)
    
    #plugins.Geocoder().add_to(m2)
    LayerControl().add_to(m2)
    
    plugins.MousePosition().add_to(m2)
    plugins.Fullscreen().add_to(m2)
    
    if 'h_canopy' in atl08_gdf.columns: atl08_gdf.drop('h_can', axis=1)
    #return (m2, atl08_gdf)
    return m2

def MAP_LAYER_FOLIUM(LAYER=None, LAYER_COL_NAME=None, fig_w=1000, fig_h=400, lat_start=60, lon_start=-120, zoom_start=8):      
    
    #Map the Layers
    Map_Figure=Figure(width=fig_w,height=fig_h)
    foliumMap = Map(
        tiles=None,
        location=(lat_start, lon_start),
        zoom_start=zoom_start, 
        control_scale=True
    )
    Map_Figure.add_child(foliumMap)
    
    if LAYER is not None:
        GEOJSON_LAYER = GeoJson(
            LAYER,
            name='footprints',
            style_function=lambda x:{'fillColor': 'gray', 'color': 'red', 'weight' : 0.75, 'opacity': 1, 'fillOpacity': 0.5},
            tooltip=features.GeoJsonTooltip(
                fields=[LAYER_COL_NAME],
                aliases=[f'{LAYER_COL_NAME}:'],
            )
        )
        #GeoJson(LAYER, name='footprints', style_function=lambda x:{'fillColor': 'gray', 'color': 'red', 'weight' : 0.75, 'opacity': 1, 'fillOpacity': 0.5}).add_to(foliumMap)
        GEOJSON_LAYER.add_to(foliumMap)
        
    basemaps['Imagery'].add_to(foliumMap)
    basemaps['basemap_gray'].add_to(foliumMap)
    basemaps['ESRINatGeo'].add_to(foliumMap)
    
    LayerControl().add_to(foliumMap)
    plugins.Geocoder().add_to(foliumMap)
    plugins.MousePosition().add_to(foliumMap)
    minimap = plugins.MiniMap()
    plugins.Fullscreen().add_to(foliumMap)
    foliumMap.add_child(minimap)
    
    return foliumMap