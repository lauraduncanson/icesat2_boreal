import geopandas
import pandas as pd
import os

import branca.colormap as cm
import matplotlib.cm

from folium import Map, TileLayer, GeoJson, LayerControl, Icon, Marker, features, Figure, CircleMarker
from folium import plugins

# Get a basemap
tiler_basemap_googleterrain = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'
tiler_basemap_gray = "http://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
tiler_basemap_image = 'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
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
        name="World gray basemap",
        attr="MAAP",
        overlay=False
    ),
    'Imagery' : TileLayer(
        tiles=tiler_basemap_image,
        opacity=1,
        name="Imagery",
        attr="MAAP",
        overlay=False
    )
}

def MAP_DPS_RESULTS(tiler_mosaic, DPS_DATA_TYPE, boreal_tile_index, tile_index_matches, tile_index_missing, 
                    mosaic_json_dict = {
                                        'agb_mosaic_json_s3_fn':    's3://maap-ops-workspace/shared/lduncanson/DPS_tile_lists/AGB_tindex_master_mosaic.json',
                                        'topo_mosaic_json_s3_fn':   's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/Topo_tindex_master_mosaic.json',
                                        'mscomp_mosaic_json_s3_fn': 's3://maap-ops-workspace/shared/nathanmthomas/DPS_tile_lists/HLS_tindex_master_mosaic.json'
                                    },
                    ecoboreal_geojson = '/projects/shared-buckets/nathanmthomas/Ecoregions2017_boreal_m.geojson',
                    max_AGB_display = 150,
                    MS_BANDNUM = 8,
                    MS_BANDMAX = 1,
                    MS_BANDCOLORBAR = 'viridis',
                    tiles_remove = [41995, 41807, 41619] # geo abyss
                   ):
    
    # Set colormaps
    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None:
        # TODO: find other valid 'colormap_names' for the tiler url that also work with cm.linear.xxxx.scale()
        agb_colormap = 'viridis'#'RdYlGn_11' #'RdYlGn' #'nipy_spectral'
        agb_tiles = f"{tiler_mosaic}?url={mosaic_json_dict['agb_mosaic_json_s3_fn']}&rescale=0,{max_AGB_display}&bidx=1&colormap_name={agb_colormap}"

        agb_se_colormap = 'magma'
        agb_se_tiles = f"{tiler_mosaic}?url={mosaic_json_dict['agb_mosaic_json_s3_fn']}&rescale=0,20&bidx=2&colormap_name={agb_se_colormap}"

        colormap_AGB = cm.linear.viridis.scale(0, max_AGB_display).to_step(25)
        colormap_AGB.caption = 'Mean of Aboveground Biomass Density [Mg/ha]'
        colormap_AGB

        #colormap_AGBSE = cm.linear.magma.scale(0, 20).to_step(5)
        #colormap_AGBSE.caption = 'Standard Error of Aboveground Biomass Density [Mg/ha]'
        #colormap = cm.linear.nipy_spectral.scale(0, 125).to_step(25)
        #colormap

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
    Map_Figure=Figure(width=1200,height=500)
    #------------------
    m1 = Map(
        #tiles="Stamen Toner",
        location=(60, 5),
        zoom_start=3, tiles=''
    )
    Map_Figure.add_child(m1)

    boreal_tiles_style = {'fillColor': '#e41a1c', 'color': '#e41a1c', 'weight' : 0.5, 'opacity': 1, 'fillOpacity': 0}
    dps_subset_style = {'fillColor': '#377eb8', 'color': '#377eb8', 'weight' : 0.75, 'opacity': 1, 'fillOpacity': 0.5}
    dps_missing_style = {'fillColor': 'red', 'color': 'red'}

    #GeoJson(atl08_gdf, name="ATL08"
    #       ).add_to(m)

    boreal_tile_index_layer = GeoJson(
            data=boreal_tile_index[~boreal_tile_index.tile_num.isin(tiles_remove)].to_crs("EPSG:4326").to_json(),
            style_function=lambda x:boreal_tiles_style,
            name="Boreal tiles",
            tooltip=features.GeoJsonTooltip(
                fields=['tile_num'],
                aliases=['Tile num:'],
            )
        )

    tile_matches_layer = GeoJson(
            data=tile_index_matches,
            style_function=lambda x:dps_subset_style,
            name=f"{DPS_DATA_TYPE} tiles completed",
            tooltip=features.GeoJsonTooltip(
                fields=['tile_num'],
                aliases=['Tile num:'],
            )
        )

    if len(tile_index_missing) > 0:
        tile_matches_missing_layer = GeoJson(
                data=tile_index_missing,
                style_function=lambda x:dps_missing_style,
                name=f"{DPS_DATA_TYPE} tiles needed"
            )

    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None:
        agb_tiles_layer = TileLayer(
            tiles=agb_tiles,
            opacity=1,
            name="Boreal AGB",
            attr="MAAP",
            overlay=True
        )
        agb_tiles_layer.add_to(m1)

        agb_se_tiles_layer = TileLayer(
            tiles=agb_se_tiles,
            opacity=1,
            name="Boreal AGB SE",
            attr="MAAP",
            overlay=True
        )
        agb_se_tiles_layer.add_to(m1)
        
    if mosaic_json_dict['mscomp_mosaic_json_s3_fn'] is not None:
        mscomp_tiles_layer = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['mscomp_mosaic_json_s3_fn']}&rescale=0.01,{MS_BANDMAX}&bidx={MS_BANDNUM}&colormap_name={MS_BANDCOLORBAR}",
            opacity=1,
            name="MS Composite",
            attr="MAAP",
            overlay=True
        )
        mscomp_tiles_layer.add_to(m1)
        
    if mosaic_json_dict['topo_mosaic_json_s3_fn'] is not None:
        topo_tiles_layer = TileLayer(
            tiles= f"{tiler_mosaic}?url={mosaic_json_dict['topo_mosaic_json_s3_fn']}&rescale=0,1&bidx=3&colormap_name=gist_gray",
            opacity=0.25,
            name="Topography",
            attr="MAAP",
            overlay=True
        )
        topo_tiles_layer.add_to(m1)


    # Add custom basemaps
    basemaps['basemap_gray'].add_to(m1)
    basemaps['Google Terrain'].add_to(m1)
    basemaps['Imagery'].add_to(m1)

    ecoboreal_layer.add_to(m1)

    # Layers are added on top. Last layer is top layer
    boreal_tile_index_layer.add_to(m1)
    tile_matches_layer.add_to(m1)

    if len(tile_index_missing) > 0:
        tile_matches_missing_layer.add_to(m1)
    #tile_matches_n_obs.add_to(m1)    
    if mosaic_json_dict['agb_mosaic_json_s3_fn'] is not None:
        m1.add_child(colormap_AGB)
    
    plugins.Geocoder().add_to(m1)
    LayerControl().add_to(m1)
    plugins.Fullscreen().add_to(m1)
    plugins.MousePosition().add_to(m1)
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
        tiles="Stamen Toner",
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
        name="ATL08 filt tiles: n_obs",
        tooltip=features.GeoJsonTooltip(
                fields=['tile_num','n_obs','local_path'],
                aliases=['Tile:','# obs.:','path:'],
            )
        )
    tile_matches_n_obs.add_to(m3)
    colormap_nobs= nobs_cmap.to_step(15)
    colormap_nobs.caption = map_name
    m3.add_child(colormap_nobs)

    LayerControl().add_to(m3)

    return m3

def map_tile_atl08(TILE_OF_INTEREST, DO_NIGHT=True):
    DPS_DATA_TYPE = 'ATL08_filt' #"Topo" "Landsat" "ATL08" "AGB"
    DPS_DATA_USER = 'lduncanson' 
    #AGB_tindex_master_fn = f's3://maap-ops-workspace/shared/{DPS_DATA_USER}/DPS_tile_lists/AGB_tindex_master.csv'
    ATL08_filt_tindex_master_fn = f'/projects/shared-buckets/{DPS_DATA_USER}/DPS_tile_lists/ATL08_filt_tindex_master.csv'
    print(ATL08_filt_tindex_master_fn)

    # Build up a dataframe from the list of dps output files
    #AGB_tindex_master = pd.read_csv(AGB_tindex_master_fn)
    #AGB_tindex_master['s3'] = [local_to_s3(local_path, user=DPS_DATA_USER, type = 'private') for local_path in AGB_tindex_master['local_path']]

    ATL08_filt_tindex_master = pd.read_csv(ATL08_filt_tindex_master_fn)
    ATL08_filt_tindex_master['s3'] = [local_to_s3(local_path, user=DPS_DATA_USER, type = 'private') for local_path in ATL08_filt_tindex_master['local_path']]

    
    # Get the CSV fn for tile
    ATL08_filt_csv_fn = ATL08_filt_tindex_master['s3'].loc[ATL08_filt_tindex_master.tile_num == TILE_OF_INTEREST].tolist()[0]
    print(ATL08_filt_csv_fn)
    
    # Get corresponding ATL08 filtered csv
    atl08_df = pd.read_csv(ATL08_filt_csv_fn)
    atl08_gdf = geopandas.GeoDataFrame(atl08_df, crs="EPSG:4326", geometry = geopandas.points_from_xy(atl08_df.lon, atl08_df.lat) )
    print(f'\nNum. of ATL08 obs. in tile {TILE_OF_INTEREST}: \t{len(atl08_gdf)}')
    print(f'Percentage of night (night_flg=1) ATL08 obs: \t\t{round(len(atl08_gdf[atl08_gdf.night_flg == 1]) / len(atl08_gdf),3) *100}%')
    print(f'Percentage of water (ValidMask=0) ATL08 obs: \t\t{round(len(atl08_gdf[atl08_gdf.ValidMask == 0]) / len(atl08_gdf),3) *100}%')
    print(f'Percentage of water (slopemask=0) ATL08 obs: \t\t{round(len(atl08_gdf[atl08_gdf.slopemask == 0]) / len(atl08_gdf),3) *100}%')

    print(round(atl08_gdf.lat.mean(),4), round(atl08_gdf.lon.mean(),4))

    # Map the Layers
    Map_Figure=Figure(width=1000,height=600)
    #------------------
    m2 = Map(
        tiles="Stamen Terrain",
        location=(atl08_gdf.lat.mean(), atl08_gdf.lon.mean()),
        zoom_start=9
    )
    Map_Figure.add_child(m2)

    if DO_NIGHT:
        atl08_gdf = atl08_gdf[atl08_gdf.night_flg == 1]
    for lat, lon, ValidMask, slopemask, h_can in zip(atl08_gdf.lat, atl08_gdf.lon, atl08_gdf.ValidMask, atl08_gdf.slopemask, atl08_gdf.h_can):
        ATL08_obs_night = CircleMarker(location=[lat, lon],
                                radius = 10,
                                weight=0.25,
                                tooltip=str(round(h_can,2))+" m",
                                fill=True,
                                #fill_color=getfill(h_can),
                                color = pal_height_cmap(h_can),
                                #color = getcolor(ValidMask),
                                opacity=1,
                                       name="ATL08 night obs"
                                
                   )

        #Map_Figure.add_child(cm)
        ATL08_obs_night.add_to(m2)

    basemaps['Imagery'].add_to(m2)
    agb_tiles_layer.add_to(m2)

    # Layera are added underneath. Last layer is bottom layer
    #boreal_tile_index_layer.add_to(m2)
    #tile_matches_missing_layer.add_to(m2)
    #tile_matches_layer.add_to(m2)

    LayerControl().add_to(m2)
    m2.add_child(pal_height_cmap)
    
    return m2