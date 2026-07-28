import geopandas
import pandas as pd
import os

import branca
import branca.colormap as cm
import matplotlib.cm

import folium
from folium import Map, TileLayer, GeoJson, LayerControl, Icon, Marker, features, Figure, CircleMarker
from folium import plugins
from folium.elements import MacroElement
from jinja2 import Template

# !pip install cogeo_mosaic
from cogeo_mosaic.mosaic import MosaicJSON
from cogeo_mosaic.backends.s3 import S3Backend
import httpx
import urllib
import json
import requests

####################
# XYZ tiles Basemaps
#
tiler_basemap_googleterrain = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'
tiler_basemap_gray =          'http://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
tiler_basemap_hillshade =     'https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}'
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
    'basemap_hillshade' : TileLayer(
        tiles=tiler_basemap_hillshade,
        opacity=1,
        name="Hillshade",
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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Correct color mapping from compute_trends.py for Kendall's Tau trend classes
cmap_colors_kendall_classes = {
    0: (1.0, 1.0, 1.0, 0.0),  # No data - transparent
    1: (0.0, 0.0, 0.8, 1.0),  # Strong sig. positive - dark blue
    2: (0.3, 0.3, 0.9, 1.0),  # Moderate sig. positive - medium blue
    3: (0.5, 0.5, 1.0, 1.0),  # Weak sig. positive - light blue
    4: (0.7, 0.7, 1.0, 1.0),  # Very weak sig. positive - very light blue
    5: (0.8, 0.8, 0.8, 1.0),  # Non-sig. positive - light gray
    6: (0.6, 0.6, 0.6, 1.0),  # Non-sig. negative - dark gray
    7: (1.0, 0.7, 0.7, 1.0),  # Very weak sig. negative - very light red
    8: (1.0, 0.5, 0.5, 1.0),  # Weak sig. negative - light red
    9: (0.9, 0.3, 0.3, 1.0),  # Moderate sig. negative - medium red
    10: (0.8, 0.0, 0.0, 1.0)  # Strong sig. negative - dark red
}

# Class labels from compute_trends.py
class_labels_kendall_classes = {
    0: 'No data',
    1: 'Strong sig. [+]',
    2: 'Mod. sig. [+]',
    3: 'Weak sig. [+]',
    4: 'Very weak sig. [+]',
    5: 'Non-sig. [+]',
    6: 'Non-sig. [-]',
    7: 'Very weak sig. [-]',
    8: 'Weak sig. [-]',
    9: 'Mod. sig. [-]',
    10: 'Strong sig. [-]'
}

def create_kendall_horizontal_legend(title="Aboveground carbon trend (ols) classes\nclassification of kendall's tau"):
    """Create a horizontal legend for Kendall's Tau trend classes with positive trends on right, negative on left."""
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Add title
    ax.text(0.5, 0.9, title, ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Reorder classes: negative trends (10,9,8,7,6), positive trends (5,4,3,2,1) - exclude class 0
    class_order = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Calculate positions for horizontal layout
    n_classes = len(class_order)
    positions = np.linspace(0.05, 0.95, n_classes)
    
    for i, class_val in enumerate(class_order):
        x_pos = positions[i]
        rgba_color = cmap_colors_kendall_classes[class_val]
        label = class_labels_kendall_classes[class_val]
        
        facecolor = rgba_color[:3]  # Use RGB, ignore alpha
        edgecolor = 'black'
        linestyle = '-'
        linewidth = 1.2
        
        # Color rectangle
        rect = patches.Rectangle((x_pos - 0.035, 0.45), 0.07, 0.25, 
                               facecolor=facecolor, 
                               edgecolor=edgecolor, 
                               linestyle=linestyle,
                               linewidth=linewidth)
        ax.add_patch(rect)
        
        # Class number - adjust text color based on background
        if class_val == 1:  # Dark blue
            text_color = 'white'
        elif class_val == 10:  # Dark red
            text_color = 'white'
        else:
            text_color = 'black'
            
        ax.text(x_pos, 0.575, f"{class_val}", 
                ha='center', va='center', 
                fontweight='bold', fontsize=16, color=text_color)
        
        # Label - split for better readability
        ax.text(x_pos, 0.3, label, 
                ha='center', va='center', 
                fontsize=12, rotation=0)
    
    # Add trend direction indicators
    ax.text(0.2, 0.05, "← Negative aboveground C trends", 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color='darkred')
    
    ax.text(0.8, 0.05, "Positive aboveground C trends →", 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color='darkblue')
    
    # Add a dashed box around the non-significant classes (6 and 5)
    # Find positions of classes 5 and 6
    class6_position = positions[class_order.index(6)]
    class5_position = positions[class_order.index(5)]
    
    # Calculate box boundaries
    box_left = class6_position - 0.045  # Left edge of class 6 box
    box_right = class5_position + 0.045  # Right edge of class 5 box
    box_bottom = 0.42  # Slightly below the color rectangles
    box_top = 0.72     # Slightly above the color rectangles
    box_width = box_right - box_left
    box_height = box_top - box_bottom
    
    # Create dashed box around non-significant classes
    nonsig_box = patches.Rectangle((box_left, box_bottom), box_width, box_height,
                                 facecolor='none', 
                                 edgecolor='black', 
                                 linestyle='--',
                                 linewidth=2,
                                 alpha=0.7)
    ax.add_patch(nonsig_box)
    
    # Add "Non-significant" label above the box
    box_center = (box_left + box_right) / 2
    ax.text(box_center, box_top + 0.05, "Non-significant", 
            ha='center', va='center', 
            fontsize=12, style='italic', alpha=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def create_classified_legend(class_dict, title="Classification"):
    """
    Create a custom legend for classified rasters.
    
    Parameters:
    -----------
    class_dict : dict
        Dictionary mapping class values to {'label': str, 'color': str}
    title : str
        Legend title
    """
    
    # Create HTML for custom legend
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>{title}</b></p>
    '''
    
    for class_val, info in class_dict.items():
        color = info['color']
        label = info['label']
        legend_html += f'''
        <p><i class="fa fa-square" style="color:{color}"></i> {label}</p>
        '''
    
    legend_html += '</div>'
    
    return legend_html

# Define your AGB trend classes with colors
AGB_TREND_CLASSES = {
    0: {'label': 'Strong decline', 'color': '#8B0000'},      # Dark red
    1: {'label': 'Moderate decline', 'color': '#DC143C'},    # Crimson
    2: {'label': 'Weak decline', 'color': '#FF6B6B'},       # Light red
    3: {'label': 'Very weak decline', 'color': '#FFB6C1'},  # Light pink
    4: {'label': 'Non-sig. decline', 'color': '#D3D3D3'},   # Light gray
    5: {'label': 'Non-sig. increase', 'color': '#C0C0C0'},  # Silver
    6: {'label': 'Very weak increase', 'color': '#98FB98'}, # Pale green
    7: {'label': 'Weak increase', 'color': '#90EE90'},      # Light green
    8: {'label': 'Moderate increase', 'color': '#32CD32'},  # Lime green
    9: {'label': 'Strong increase', 'color': '#006400'},    # Dark green
    10: {'label': 'Very strong increase', 'color': '#004225'} # Very dark green
}

def make_tiles_layer_dict_with_custom_legend(mosaic_reg_id, mosaic_json_fn, NAME: str, 
                                           class_dict: dict, SHOW_CBAR=True, 
                                           PARAMS_DICT=None, PRINT=False):
    """
    Enhanced version that creates tiles with custom legend for classified data.
    """
    
    if PARAMS_DICT is None:
        PARAMS_DICT = {"rescale": "0,10", "bidx": "1", "colormap_name": 'RdBu_r'}
    
    # Build tiles URL
    tiles = build_tiles_with_params(mosaic_reg_id, mosaic_json_fn, 
                                  params_dict=PARAMS_DICT, 
                                  titiler_endpoint="https://titiler.maap-project.org")
    
    # Create the tile layer
    tiles_layer = folium.TileLayer(
        tiles=tiles,
        opacity=1,
        name=NAME,
        attr="MAAP",
        overlay=True
    )
    
    # Create custom legend if requested
    custom_legend = None
    if SHOW_CBAR and class_dict:
        custom_legend = create_classified_legend(class_dict, NAME)
    
    return {
        'layer': tiles_layer,
        'legend_html': custom_legend,
        'caption': NAME,
        'show_cbar': SHOW_CBAR,
        'cmap': 'RdBu_r', # qwik fix
        'min_val': 0, # qwik fix
        'max_val': 10 # qwik fix
    }
    
def local_to_s3(url, user = 'nathanmthomas', type='public'):
    ''' A Function to convert local paths to s3 urls'''
    if type == 'public':
        replacement_str = f's3://maap-ops-workspace/shared/{user}'
    else:
        replacement_str = f's3://maap-ops-workspace/{user}'
    return url.replace(f'/projects/my-{type}-bucket', replacement_str)

######################
# Functions to prepare tile layer dictionaries needed for mapping
#
def register_mosaic_json_titiler(mosiac_json_fn, titiler_endpoint, print_info=False):
    '''
    Mosaic registration that is specific to TiTiler
    '''
    if isinstance(mosiac_json_fn, list):
        # a list of s3 links
        # this is slow!
        mosaicdata = MosaicJSON.from_urls(mosiac_json_fn)
        
    else:
        with S3Backend(mosiac_json_fn) as mosaic:
            mosaicdata = mosaic.mosaic_def

    mosaic_links = httpx.post(
                        url=f"{titiler_endpoint}/mosaics",
                        headers={
                            "Content-Type": "application/vnd.titiler.mosaicjson+json",
                        },
                        json=mosaicdata.model_dump(exclude_none=True),
                    ).json()
    
    if print_info: 
        print(mosaic_links)
        
    return mosaic_links

def build_tiles_with_params(mosaic_reg_id, mosiac_json_fn, params_dict, titiler_endpoint = "https://titiler.maap-project.org", DEBUG=True):
    
    '''
    Identifies or generates a mosaic json registration id with TiTiler (use mosaiclib.TITILER_MOSAIC_REG_DICT[TYPE][YEAR])
    Returns a tiles layer for folium
    '''
    if mosaic_reg_id is None and mosiac_json_fn is None:
        print('Need either a mosaic registration id or a valid mosaic json filename.\nExiting.')
        sys.exit(1)
    
    if mosaic_reg_id is None:

        # colormap_name needs to be .lower() here
        params_dict_tmp = params_dict.copy()
        if 'colormap_name' in params_dict:
            params_dict_tmp['colormap_name'] = params_dict_tmp['colormap_name'].lower()
        
        if DEBUG: print(f'Registering mosaic with Titiler: {mosiac_json_fn}')
        
        mosaic_links = register_mosaic_json_titiler(mosiac_json_fn, titiler_endpoint, print_info=False)
        tilejson_endpoint = list(
                                filter(lambda x: x.get("rel") == "tilejson", dict(mosaic_links)["links"])
                                )
        r_te_json = httpx.get(tilejson_endpoint[0]["href"], params=params_dict_tmp).json()
        if DEBUG: print(r_te_json)
            
        tiles = f"{r_te_json['tiles'][0]}"
        if 'colormap_name' in params_dict:
            tiles = tiles.replace(params_dict_tmp['colormap_name'], params_dict['colormap_name'])
    else:
        tiles = "".join([titiler_endpoint, "/mosaics/", mosaic_reg_id, "/tiles/{z}/{x}/{y}", "@1x?", urllib.parse.urlencode(params_dict)])
    
    return tiles

def make_tiles_layer_dict(mosaic_reg_id, mosaic_json_fn, NAME: str, SHOW_CBAR=False, SHOW_LAYER=False, PARAMS_DICT = {"rescale": "0,30", "bidx":"1", "colormap_name": "inferno"}, PRINT=False):
    
    '''
    Use mosaic json to check for registration, register, build tiles layer url with parameters
    A NAME will help identify the tiles layer in the legend and under the colorbar
    '''
    if PRINT: print(f'\n\n{PARAMS_DICT}\n\n')
    # if type(mosaic_json_fn) is list:
    #     mosaic_links = register_mosaic_json_titiler(mosaic_json_fn)
    tiles = build_tiles_with_params(mosaic_reg_id, mosaic_json_fn, params_dict = PARAMS_DICT, titiler_endpoint = "https://titiler.maap-project.org")
    MIN, MAX = eval(PARAMS_DICT["rescale"])
    if "colormap_name" in PARAMS_DICT:
        if PRINT: print(f'\n\n{PARAMS_DICT}\n\n')
        CMAP = PARAMS_DICT['colormap_name']
        if PRINT: print(f'CMAP unaltered: {CMAP}')
        tiles = tiles.split('rescale')[0] + f'rescale={MIN},{MAX}' + f"&bidx={PARAMS_DICT['bidx']}&colormap_name={CMAP.lower()}"
        if PRINT: print(f'tiles with CMAP changed: {tiles}')
        tiles_layer  = TileLayer(
            #tiles= f"{tiler_mosaic}?url={mosaic_json_fn}&rescale={MIN},{MAX}&bidx={BANDNUM}&colormap_name={CMAP}",
            #tiles= f"{tiler_mosaic}&rescale={MIN},{MAX}&bidx={BANDNUM}&colormap_name={CMAP}",
            tiles = tiles,
            opacity=1,
            name=NAME,
            attr="MAAP",
            overlay=True,
            show=SHOW_LAYER
        )
    else:
        print('Custom colormap...')
        if False:
            #CMAP = PARAMS_DICT['colormap'].split(',')
            CMAP = cm.LinearColormap(colors = PARAMS_DICT['colormap'].split(','), vmin=float(PARAMS_DICT['rescale'].split(',')[0]), vmax=float(PARAMS_DICT['rescale'].split(',')[1]))#.to_step(n=len(forest_height_colors))
            rgbas = [[int(value) for value in rgb] for rgb in CMAP.to_rgba(x=bins[:-1], bytes=True)]
            
    #         # Make sure its json serializable# https://chatgpt.com/c/66f40d2b-7594-800a-9940-3dd8c09b7ce0
    #         # Create a function to convert the colormap to a step list that Folium can use
    #         def create_color_stops(colormap, n=10):
    #             return [(colormap.rgb_hex_str(x), x / (n - 1)) for x in range(n)]

    #         # Convert the colormap to a list of color stops
    #         color_stops = create_color_stops(CMAP, len(PARAMS_DICT['colormap'].split(',')))

            # take colormap out of tiles string and use it explicitly in TileLayer def below
            tiles = tiles.split('&colormap')[0]
            tiles_layer  = TileLayer(
                tiles = tiles,
                opacity=1,
                name=NAME,
                attr="MAAP",
                overlay=True,
            #colormap=lambda x: color_stops(x)
            #colormap=color_stops
            )
        else:
            CMAP = PARAMS_DICT['colormap']
            tiles_layer  = TileLayer(
                tiles = tiles,
                opacity=1,
                name=NAME,
                attr="MAAP",
                overlay=True,
            )

    TILES_LAYER_DICT = {
        "layer": tiles_layer,
        "cmap": CMAP,
        "min_val": MIN,
        "max_val": MAX, #float(PARAMS_DICT['rescale'].split(',')[1]),
        "caption": tiles_layer.layer_name,
        "show_cbar": SHOW_CBAR
    }
    
    if PRINT:
        print(f'{NAME} tiles ({CMAP}): {tiles}')
    
    return TILES_LAYER_DICT

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

def MAP_REGISTERED_DPS_RESULTS( 
                    boreal_tile_index, 
                    tile_index_check=None, 
                    CHECK_TILES_NAME=None, 
                    ecoboreal_geojson = '/projects/shared-buckets/nathanmthomas/analyze_agb/input_zones/wwf_circumboreal_Dissolve.geojson',
                    tiles_remove = [None], # geo abyss,
                    SHOW_WIDGETS=False,
                    map_width=1000, map_height=500,
                    ADD_TILELAYER = None,
                    ADD_GEOJSONLAYER = None,
                    ADD_4326_GPKG = None,
                    enable_pixel_query=True, raster_query_configs=None
                   ):
    
    if ADD_TILELAYER is not None:
        if isinstance(ADD_TILELAYER, list):
            ADD_TILELAYER_LIST = ADD_TILELAYER
        else:
            ADD_TILELAYER_LIST = [ADD_TILELAYER]
            
        colormap_ADDED_TILELAYER_list = []
        for ADD_TILELAYER in ADD_TILELAYER_LIST:
            cmap = matplotlib.cm.get_cmap(ADD_TILELAYER["cmap"], 25)
            colormap_ADDED_TILELAYER = branca.colormap.LinearColormap(colors=[matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]).scale(ADD_TILELAYER["min_val"], ADD_TILELAYER["max_val"])
            colormap_ADDED_TILELAYER.caption = ADD_TILELAYER["caption"]
            colormap_ADDED_TILELAYER_list.append(colormap_ADDED_TILELAYER)

    if ADD_GEOJSONLAYER is not None:
        if isinstance(ADD_GEOJSONLAYER, list):
            ADD_GEOJSONLAYER_LIST = ADD_GEOJSONLAYER
        else:
            ADD_GEOJSONLAYER_LIST = [ADD_GEOJSONLAYER]

    # Style Vector Layers
    ecoboreal_style = {'fillColor': 'gray', 'color': 'gray'}
    boreal_style = {'fillColor': 'gray', 'color': 'gray'}
    boreal_subset_style = {'fillColor': 'red', 'color': 'red'}

    if ecoboreal_geojson is not None:

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

    if ADD_4326_GPKG is not None:
        if isinstance(ADD_4326_GPKG, list):
            ADD_4326_GPKG_LIST = ADD_4326_GPKG
        else:
            ADD_4326_GPKG_LIST = [ADD_4326_GPKG]
            
        GEOJSON_FROM_ADD_4326_GPKG_LIST = []
        for DICT_4326_GPKG in ADD_4326_GPKG_LIST:
            gdf = geopandas.read_file(DICT_4326_GPKG['fn'])
            GEOJSON_FROM_ADD_4326_GPKG_LIST.append(GeoJson(gdf, 
                                                           name=DICT_4326_GPKG['desc'], 
                                                           style_function=lambda x:DICT_4326_GPKG['style'],
                                                          ))

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

    boreal_tile_index_layer = GET_BOREAL_TILE_LAYER(boreal_tile_index, tiles_remove, boreal_tiles_style)

    if tile_index_check is not None and len(tile_index_check) > 0:
        tile_index_check_layer = GeoJson(
                data=tile_index_check,
                style_function=lambda x:dps_check_style,
                name=f"{CHECK_TILES_NAME} tiles"
            ) 

    if ADD_TILELAYER is not None:
        for i, ADD_TILELAYER in enumerate(ADD_TILELAYER_LIST):
            ADD_TILELAYER["layer"].add_to(m1)
            print(f"Adding layer {ADD_TILELAYER['caption']}...")
            if ADD_TILELAYER["show_cbar"]:
                m1.add_child(colormap_ADDED_TILELAYER_list[i])

    if ADD_4326_GPKG is not None:
        for i, ADD_GEOJSON_FROM_GPKG in enumerate(GEOJSON_FROM_ADD_4326_GPKG_LIST):
            ADD_GEOJSON_FROM_GPKG.add_to(m1)

    if ADD_GEOJSONLAYER is not None:
        for i, ADD_GEOJSONLAYER in enumerate(ADD_GEOJSONLAYER_LIST):
            ADD_GEOJSONLAYER.add_to(m1)
        
        
    # Add custom basemaps
    basemaps['Google Terrain'].add_to(m1)
    basemaps['Imagery'].add_to(m1)
    basemaps['ESRINatGeo'].add_to(m1)
    basemaps['basemap_gray'].add_to(m1)
    basemaps['basemap_hillshade'].add_to(m1)

    if ecoboreal_geojson is not None:
        ecoboreal_layer.add_to(m1)

    # Layers are added on top. Last layer is top layer
    boreal_tile_index_layer.add_to(m1)

    if tile_index_check is not None and len(tile_index_check) > 0:
        tile_index_check_layer.add_to(m1) 
    
    if SHOW_WIDGETS:
        plugins.Geocoder().add_to(m1)

    m1 = MAP_CONTROL(m1)
    # LayerControl().add_to(m1)
    # plugins.Geocoder(position='bottomright').add_to(m1)
    # plugins.Fullscreen(position='bottomleft').add_to(m1)
    # plugins.MousePosition().add_to(m1)
    
    if SHOW_WIDGETS:
        minimap = plugins.MiniMap()
        m1.add_child(minimap)
        #m1.add_child(colormap_AGBSE)

    # Add pixel query tool if requested
    if enable_pixel_query and raster_query_configs:
        m1 = add_advanced_pixel_query_tool(m1, raster_query_configs)
        m1 = add_coordinate_display(m1)
    
    return m1
    
def MAP_CONTROL(m):
    LayerControl().add_to(m)
    plugins.Geocoder(position='bottomright').add_to(m)
    plugins.Fullscreen(position='bottomleft').add_to(m)
    plugins.MousePosition().add_to(m)
    return m

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

def add_advanced_pixel_query_tool(folium_map, raster_configs):
    """
    Add advanced pixel query tool that can query multiple rasters.
    
    Parameters:
    -----------
    folium_map : folium.Map
        The folium map object
    raster_configs : list
        List of dicts with raster configuration:
        [{'name': 'Carbon', 'url': 'titiler_url', 'band': 1}, ...]
    """
    
    # Create JavaScript configuration
    raster_js_config = json.dumps(raster_configs)
    
    pixel_query_js = f"""
    <script>
    var rasterConfigs = {raster_js_config};
    
    // Add click event listener
    {folium_map.get_name()}.on('click', function(e) {{
        var lat = e.latlng.lat;
        var lng = e.latlng.lng;
        
        var popupContent = `
            <div style="font-family: Arial, sans-serif; min-width: 250px;">
                <h4>Pixel Query</h4>
                <p><strong>Coordinates:</strong> ${{lat.toFixed(6)}}, ${{lng.toFixed(6)}}</p>
                <div id="pixel-values">Querying...</div>
            </div>
        `;
        
        var popup = L.popup()
            .setLatLng(e.latlng)
            .setContent(popupContent)
            .openOn({folium_map.get_name()});
        
        // Query all configured rasters
        var promises = rasterConfigs.map(config => {{
            var queryUrl = `${{config.url}}/point/${{lng}},${{lat}}?bidx=${{config.band}}`;
            return fetch(queryUrl)
                .then(response => response.json())
                .then(data => ({{
                    name: config.name,
                    value: data.values ? data.values[0] : 'No data',
                    band: config.band
                }}))
                .catch(error => ({{
                    name: config.name,
                    value: 'Error',
                    band: config.band
                }}));
        }});
        
        Promise.all(promises).then(results => {{
            var valuesHtml = '<table style="width: 100%; border-collapse: collapse;">';
            valuesHtml += '<tr><th style="border: 1px solid #ddd; padding: 5px;">Layer</th>';
            valuesHtml += '<th style="border: 1px solid #ddd; padding: 5px;">Value</th>';
            valuesHtml += '<th style="border: 1px solid #ddd; padding: 5px;">Band</th></tr>';
            
            results.forEach(result => {{
                valuesHtml += `<tr>
                    <td style="border: 1px solid #ddd; padding: 5px;">${{result.name}}</td>
                    <td style="border: 1px solid #ddd; padding: 5px;">${{result.value}}</td>
                    <td style="border: 1px solid #ddd; padding: 5px;">${{result.band}}</td>
                </tr>`;
            }});
            
            valuesHtml += '</table>';
            
            document.getElementById('pixel-values').innerHTML = valuesHtml;
        }});
    }});
    </script>
    """
    
    # Add the JavaScript to the map
    folium_map.get_root().html.add_child(folium.Element(pixel_query_js))
    
    return folium_map

def add_coordinate_display(folium_map):
    """Add coordinate display to folium map."""
    
    coordinate_js = f"""
    <div id="coordinate-display" style="
        position: fixed; 
        bottom: 10px; 
        left: 10px; 
        background: rgba(255,255,255,0.9); 
        padding: 5px 10px; 
        border-radius: 3px; 
        font-family: monospace; 
        font-size: 12px;
        z-index: 1000;
        border: 1px solid #ccc;">
        Lat: -, Lon: -
    </div>
    
    <script>
    {folium_map.get_name()}.on('mousemove', function(e) {{
        var lat = e.latlng.lat.toFixed(6);
        var lng = e.latlng.lng.toFixed(6);
        document.getElementById('coordinate-display').innerHTML = 
            `Lat: ${{lat}}, Lon: ${{lng}}`;
    }});
    
    {folium_map.get_name()}.on('mouseout', function(e) {{
        document.getElementById('coordinate-display').innerHTML = 
            'Lat: -, Lon: -';
    }});
    </script>
    """
    
    folium_map.get_root().html.add_child(folium.Element(coordinate_js))
    return folium_map

def add_pixel_query(m, query_layers, decimals=3):
    """
    Add click-to-query functionality to a folium map.

    Each layer in `query_layers` is a dict with:
        'name'      : display label (required)
        'point_url' : URL with {lon},{lat} placeholders (required)
        'units'     : optional unit suffix
        'section'   : optional group heading; consecutive layers sharing
                      the same section appear under one heading
    """
    layers_json = json.dumps(query_layers)
    js = """
    {% macro script(this, kwargs) %}
    (function() {
        var map = {{ this._parent.get_name() }};
        var queryLayers = {{ this.layers_json }};
        var decimals = {{ this.decimals }};

        map.on('click', async function(e) {
            var lat = e.latlng.lat;
            var lon = e.latlng.lng;

            var popup = L.popup({maxWidth: 420})
                .setLatLng(e.latlng)
                .setContent('<b>Loading…</b>')
                .openOn(map);

            // Fetch all values in parallel, preserving original index order
            var results = await Promise.all(queryLayers.map(async function(layer, idx) {
                var url = layer.point_url
                    .replace('{lon}', lon.toFixed(6))
                    .replace('{lat}', lat.toFixed(6));
                var entry = {
                    idx: idx,
                    section: layer.section || null,
                    name: layer.name,
                    units: layer.units || ''
                };
                try {
                    var resp = await fetch(url);
                    if (!resp.ok) { entry.value = 'http ' + resp.status; return entry; }
                    var txt = await resp.text();
                    txt = txt.replace(/:\s*NaN/g, ': null')
                             .replace(/:\s*-?Infinity/g, ': null');
                    var data = JSON.parse(txt);
                    var v = null;
                    if (data.values && data.values.length) {
                        var first = data.values[0];
                        if (Array.isArray(first)) {
                            v = (first.length >= 2 && Array.isArray(first[1]) && first[1].length)
                                ? first[1][0] : null;
                        } else {
                            v = first;
                        }
                    }
                    if (v === null || v === undefined || Number.isNaN(v) || !Number.isFinite(v)) {
                        entry.value = 'no data';
                    } else {
                        entry.value = Number(v).toFixed(decimals) + (entry.units ? ' ' + entry.units : '');
                    }
                } catch (err) {
                    entry.value = 'error';
                }
                return entry;
            }));

            // Build HTML, grouping consecutive entries by section
            var html = '<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 12px; min-width: 280px;">';
            html += '<div style="border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-bottom: 8px;">';
            html += '<b style="color:#1f4d8c;">Pixel values</b><br>';
            html += '<span style="color:#888; font-size: 0.95em;">lat ' + lat.toFixed(5) +
                    ', lon ' + lon.toFixed(5) + '</span></div>';

            var lastSection = '__init__';
            results.forEach(function(r) {
                if (r.section !== lastSection) {
                    if (lastSection !== '__init__') {
                        html += '</table>';   // close previous group
                    }
                    if (r.section) {
                        html += '<div style="font-weight:600; color:#1f4d8c; ' +
                                'margin: 8px 0 4px 0; font-size: 0.95em; ' +
                                'text-transform: uppercase; letter-spacing: 0.03em;">' +
                                r.section + '</div>';
                    } else {
                        html += '<div style="height: 8px;"></div>';   // spacer for ungrouped
                    }
                    html += '<table style="border-collapse:collapse; width:100%;">';
                    lastSection = r.section;
                }
                html += '<tr>' +
                        '<td style="padding: 1px 12px 1px 0; color:#555;">' + r.name + '</td>' +
                        '<td style="font-family: monospace; padding: 1px 0; text-align:right;">' +
                        '<b>' + r.value + '</b></td>' +
                        '</tr>';
            });
            html += '</table></div>';

            popup.setContent(html);
        });
    })();
    {% endmacro %}
    """
    el = MacroElement()
    el._template = Template(js)
    el.layers_json = layers_json
    el.decimals = decimals
    m.add_child(el)
    return m