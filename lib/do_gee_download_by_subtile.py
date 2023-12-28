import os
import subprocess
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ee
from shapely.geometry import Polygon, box
import contextily as ctx
from shapely import geometry
import ee_download
import argparse
import zipfile
import glob
import rasterio
from osgeo import gdal

def get_gee_assets(asset_path):
    
    # fetch EE assets
    COLL_ID = asset_path
    collection = ee.ImageCollection(COLL_ID)
    collection_items = collection.toList(1000).getInfo()
    # tranlsate collection to df
    collection_items_df = pd.DataFrame.from_dict(collection_items)

    return collection_items_df

def create_polygon_from_coordinates(coordinates, crs='EPSG:4326'):
    """
    Create a polygon from a list of input coordinates and return it as a GeoDataFrame.

    Parameters:
    - coordinates (list of tuples): List of input coordinates (e.g., [(x1, y1), (x2, y2), ...]).
    - crs (str): Coordinate reference system. Default is EPSG:4326 (WGS84).

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing the created polygon.
    """
    # Create a Shapely Polygon from the input coordinates
    polygon = Polygon(coordinates)

    # Create a GeoDataFrame with the polygon
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)

    return gdf

#def create_fishnet(collection_items_df, asset_num, dims, DEBUG=False):
def create_fishnet(asset, dims, DEBUG=False, 
                   IN_CRS = 'PROJCS["unnamed",GEOGCS["GRS 1980(IUGG, 1980)",DATUM["unknown",SPHEROID["GRS80",6378137,298.257222101],TOWGS84[0,0,0,0,0,0,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",50],PARAMETER["standard_parallel_2",70],PARAMETER["latitude_of_center",40],PARAMETER["longitude_of_center",180],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
                      ):
    
    try:
        # Replace tile loc with 
        #asset = collection_items_df.iloc[asset_num]
        
        asset = pd.concat([asset.drop(['bands'], axis=1), asset['bands'].apply(pd.Series)], axis=1)
        asset = pd.concat([asset.drop([0], axis=1), asset[0].apply(pd.Series)], axis=1)

        #rows = asset['bands'][0][0]['dimensions'][0]
        #cols = asset['bands'][0][0]['dimensions'][1]
        rows, cols = asset['dimensions'].iloc[0]
    
        if False:
            print(f'\tasset id = {list(asset.id.iloc[0])[0]} (rows={rows}, cols={cols})')
        #print('\trows =', rows, 'cols = ', cols)
        if DEBUG:
            print(asset['crs_transform'])
        #minx = (asset['bands'][0][0]['crs_transform'][2])
        #miny = (asset['bands'][0][0]['crs_transform'][5])
        minx = asset['crs_transform'].iloc[0][2]
        miny = asset['crs_transform'].iloc[0][5]
        res = asset['crs_transform'].iloc[0][0]
        maxx = minx + (cols*res)
        maxy = miny - (rows*res)
    
        if DEBUG:
            print('\tbbox Albers=', minx, miny, maxx, maxy)

        #create a fishnet: https://spatial-dev.guru/2022/05/22/create-fishnet-grid-using-geopandas-and-shapely/
        gdf_tile = gpd.GeoDataFrame(index=[0],geometry=[Polygon([[minx,miny],[minx,maxy],[maxx,maxy],[maxx, miny]])], crs=IN_CRS)
    
        # Get the extent of the shapefile
        minX, minY, maxX, maxY = gdf_tile.total_bounds

        # Create a fishnet
        x, y = (minX, minY)
        geom_array = []

        # Polygon Size
        square_size = dims*res
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size

        # Convert fishnet to gdf
        fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs(IN_CRS)

    except Exception as e:
        raise e
    
    return fishnet

def create_fishnet_new(gdf, cell_size_meters, tile_col_name, subtile_col_name='subtile_num', FIX_DATELINE=False, 
                       IN_CRS = 'PROJCS["unnamed",GEOGCS["GRS 1980(IUGG, 1980)",DATUM["unknown",SPHEROID["GRS80",6378137,298.257222101],TOWGS84[0,0,0,0,0,0,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",50],PARAMETER["standard_parallel_2",70],PARAMETER["latitude_of_center",40],PARAMETER["longitude_of_center",180],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
                      ):
    """
    Create a fishnet of vector polygons that fully cover each polygon in the input GeoDataFrame.

    Parameters:
    - gdf (geopandas.GeoDataFrame): Input GeoDataFrame with polygons.
    - cell_size_meters (float): Size of each grid cell in meters.

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing the fishnet polygons.
    """
    
    # Input GDF probably in GCS - fishnet needs it in projected coord systems
    gdf = gdf.to_crs(IN_CRS)
    
    # Ensure the GeoDataFrame has a valid geometry column
    if 'geometry' not in gdf.columns or gdf['geometry'].isna().any():
        raise ValueError("The GeoDataFrame must have a valid 'geometry' column with polygons.")

    fishnet_gdf_list = []
    
    # Iterate through each polygon in the input GeoDataFrame
    for _, row in gdf.iterrows():
        polygon = row['geometry']
        tile_num = row[tile_col_name]
        # Get the bounding box of the polygon and extend it to ensure full coverage
        bounds = polygon.bounds
        extended_bounds = (
            bounds[0] - cell_size_meters,
            bounds[1] - cell_size_meters,
            bounds[2] + cell_size_meters,
            bounds[3] + cell_size_meters
        )

        # Create a fishnet for the current polygon
        fishnet_cells = create_fishnet_cells(extended_bounds, cell_size_meters)

        # Append the fishnet cells to the fishnet GeoDataFrame
        fishnet_gdf_tmp = gpd.GeoDataFrame({tile_col_name: [tile_num for i in range(len(fishnet_cells))], subtile_col_name: [i for i in range(len(fishnet_cells))]}, geometry=fishnet_cells, crs=gdf.crs)
        if FIX_DATELINE:
            # Fix dateline
            fishnet_gdf_tmp = split_polygons_by_dateline_poly(fishnet_gdf_tmp, tile_num_col=subtile_col_name)
        fishnet_gdf_tmp[tile_col_name] = tile_num # this converts NULL to the actual tile num for the new polys resulting from the split
        fishnet_gdf_list.append(fishnet_gdf_tmp)
        
    fishnet_gdf = pd.concat(fishnet_gdf_list)
    return fishnet_gdf

def create_fishnet_cells(bounds, cell_size):
    """
    Create fishnet cells for a given bounding box and cell size.

    Parameters:
    - bounds (tuple): Bounding box coordinates (minx, miny, maxx, maxy).
    - cell_size (float): Size of each grid cell.

    Returns:
    - List of shapely.geometry.Polygon: List of fishnet cells.
    """
    minx, miny, maxx, maxy = bounds
    rows = int((maxy - miny) / cell_size)
    cols = int((maxx - minx) / cell_size)

    fishnet_cells = []
    for i in range(rows):
        for j in range(cols):
            x = minx + j * cell_size
            y = miny + i * cell_size
            cell = Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)])
            fishnet_cells.append(cell)

    return fishnet_cells

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, LineString
from shapely.ops import unary_union

def create_date_line_polygon(buffer_distance=0.00001):
    """
    Create a polygon representing the international date line buffered by a given distance.

    Parameters:
    - buffer_distance (float): Distance to buffer the international date line. Default is 0.00005 dd - about 5m in the boreal.

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame with the buffered international date line polygon.
    """
    # Create a LineString representing the international date line
    date_line = LineString([(-180, -89.99), (-180, 89.99)])

    # Buffer the line to create a polygon
    date_line_polygon = date_line.buffer(buffer_distance)

    # Create a GeoDataFrame with the buffered polygon
    gdf_date_line = gpd.GeoDataFrame(geometry=[date_line_polygon], crs='EPSG:4326')

    return gdf_date_line

def split_polygon(polygon, crs):
    """
    Split a polygon by the international date line.

    Parameters:
    - polygon (shapely.geometry.Polygon): Polygon to be split.

    Returns:
    - List of shapely.geometry.Polygon: List of split polygons.
    """
    date_line = create_date_line_polygon().to_crs(crs).geometry[0]
    
    parts = []
    if polygon.intersects(date_line):
        # Split the polygon
        parts = list(polygon.difference(date_line).geoms) 
    else:
        parts.append(polygon)

    return parts

def split_polygons_by_dateline_poly(gdf, tile_num_col='tile_num', DO_BOREAL_TILES_COLS=False):
    """
    Split polygons in a GeoDataFrame by the international date line.

    Parameters:
    - gdf (geopandas.GeoDataFrame): GeoDataFrame with polygons to be split.

    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame with split polygons.
    """
    # Ensure the GeoDataFrame has a valid geometry column
    if 'geometry' not in gdf.columns or gdf['geometry'].isna().any():
        raise ValueError("The GeoDataFrame must have a valid 'geometry' column with polygons.")

    # Split polygons by the international date line
    split_polygons = []
    centroid_gdf_list = []
    tile_num_list = []
    if 'tile_group' in gdf.columns and 'tile_version' in gdf.columns: DO_BOREAL_TILES_COLS = True
    if DO_BOREAL_TILES_COLS:
        tile_group_list = []
        tile_version_list = []
    for _, row in gdf.iterrows():
        original_polygon = row['geometry']
        tile_num = row[tile_num_col]
        centroid_gdf = gpd.GeoDataFrame(data=[row], geometry=[row['geometry'].centroid], crs=gdf.crs) # this will help carry through attributes
        if original_polygon.geom_type == 'Polygon':
            # Handle a single Polygon
            split_parts = split_polygon(original_polygon, crs=gdf.crs)
        elif original_polygon.geom_type == 'MultiPolygon':
            # Handle MultiPolygon
            split_parts = [split_polygon(part, crs=gdf.crs) for part in original_polygon]
        else:
            raise ValueError(f"Unsupported geometry type: {original_polygon.geom_type}")
        #print(len(split_parts))
        split_polygons.extend(split_parts)
        centroid_gdf_list.append(centroid_gdf)
        for part in split_parts:
            if tile_num in tile_num_list:
                # New split polys are scaled by 100 from original tile_num values
                tile_num = tile_num * 100
            #print(tile_num)
            tile_num_list.append(tile_num)
            
            if DO_BOREAL_TILES_COLS:
                tile_group_list.append(row['tile_group'])
                tile_version_list.append(row['tile_version'])

    # Create a new GeoDataFrame with split polygons
    gdf_split = gpd.GeoDataFrame(geometry=split_polygons, crs=gdf.crs)
    gdf_split = gpd.sjoin(gdf_split, pd.concat(centroid_gdf_list), how='left')
    print(f'Length of tile num list: {len(tile_num_list)}')
    ###gdf_split[tile_num_col] = tile_num_list
    if DO_BOREAL_TILES_COLS:
        gdf_split['tile_group'] = tile_group_list
        gdf_split['tile_version'] = tile_version_list
    return gdf_split



import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
import numpy as np
import rasterio
from CovariateUtils import write_cog, get_index_tile, get_shape, reader

def create_multiband_geotiff(output_path, band_names, single_band_geotiffs):
    """
    Create a multiband geotiff from a list of single band geotiffs.

    Parameters:
    - output_path (str): Path to the output multiband geotiff file.
    - band_names (list): List of band names corresponding to the single band geotiffs.
    - single_band_geotiffs (list): List of paths to single band geotiffs.

    Returns:
    - None
    """

    # Open the first single band geotiff to get metadata
    with rasterio.open(single_band_geotiffs[0]) as src:
        profile = src.profile.copy()
        profile.update(count=len(band_names),
                       crs=src.profile['crs'], interleave='band',
                       height=src.profile['height'], width=src.profile['width'], dtype=src.profile['dtype']
                       #, nodata=-9999.0
                      )
    profile['blockxsize']=256
    profile['blockysize']=256
    profile['predictor']=1 # originally set to 2, which fails with 'int'; 1 tested successfully for 'int' and 'float64'
    profile['zlevel']=7
    profile['tiled']=True
    profile['compress']='deflate'

    # Create a new multiband geotiff
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, band_name in enumerate(band_names, 1):
            # Ensure that the single band geotiffs have the same shape, data type, and projection
            with rasterio.open(single_band_geotiffs[i-1]) as src_band:
                if src_band.shape[0] != profile['height'] or src_band.shape[1] != profile['width']:
                    raise ValueError(f"Shape of {single_band_geotiffs[i-1]} does not match the expected shape.")

                # Read and write each band to the new multiband geotiff
                dst.write(src_band.read(1), indexes=i)
                dst.set_band_description(i, band_name)

    print(f"Multiband geotiff created successfully at: {output_path}")

def create_multiband_cog(output_path, band_names, single_band_geotiffs, tile_gdf, input_nodata_value: float):
    """
    Create a multiband COG from a list of single band geotiffs.

    Parameters:
    - output_path (str): Path to the output multiband geotiff file.
    - band_names (list): List of band names corresponding to the single band geotiffs.
    - single_band_geotiffs (list): List of paths to single band geotiffs.

    Returns:
    - None
    """

    # Open the first single band geotiff to get metadata
    with rasterio.open(single_band_geotiffs[0]) as src:
        profile = src.profile.copy()
        profile.update(count=len(band_names),
                       crs=src.profile['crs'], interleave='band',
                       height=src.profile['height'], width=src.profile['width'], dtype=src.profile['dtype']
                       #, nodata=-9999.0
                      )
    profile['blockxsize']=256
    profile['blockysize']=256
    profile['predictor']=1 # originally set to 2, which fails with 'int'; 1 tested successfully for 'int' and 'float64'
    profile['zlevel']=7
    profile['tiled']=True
    profile['compress']='deflate'

    # Create a new multiband geotiff
    array_list = []
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, band_name in enumerate(band_names, 1):
            # Ensure that the single band geotiffs have the same shape, data type, and projection
            with rasterio.open(single_band_geotiffs[i-1]) as src_band:
                if src_band.shape[0] != profile['height'] or src_band.shape[1] != profile['width']:
                    raise ValueError(f"Shape of {single_band_geotiffs[i-1]} does not match the expected shape.")

                # Read and write each band to the new multiband geotiff
                dst.write(src_band.read(1), indexes=i)
                dst.set_band_description(i, band_name)
                # Build array list for use in write_cog()
                array_list.append(src_band.read(1))
    # Stack
    # move axis of the stack so bands is first
    stack = np.transpose(array_list, [0,1,2])

    if True:
        print('\nApply a common mask across all layers of stack...')
        print(f'Masking out the input nodata value {input_nodata_value}...')
        print(f"Stack shape pre-mask:\t\t{stack.shape}")
        stack[:, np.any(stack == input_nodata_value, axis=0)] = input_nodata_value 
        print('Masking out 0 value (appears along some edges of S1 subtiles...')
        stack[:, np.any(stack == 0, axis=0)] = input_nodata_value
        print(f"Stack shape post-mask:\t\t{stack.shape}")

    
    # TODO: to be safe out_crs should be read from the gdal dataset
    if os.path.exists(output_path):
        print('Removing existing geotiff before writing COG version...')
        os.remove(output_path)
    write_cog(stack, 
              output_path, 
              profile['crs'], 
              profile['transform'],
              band_names, 
              # clip_geom = tile_gdf,      # Using this caused memory crashes (maybe feeding wrong gdf?) not needed now anyway.
              # clip_crs = profile['crs'],
              # resolution = (res, res),
              input_nodata_value = input_nodata_value,
              align = False)               # Keep as False so clip_geom not needed

    print(f"Multiband COG created successfully at: {output_path}")

# TODO: use an id field 'AGG_TILE_NUM'
# replace TILE_LOC with id number of id field
def do_gee_download_by_subtile(SUBTILE_LOC, 
                               #TILELOC,
                               ID_NUM,
                               ID_COL,
                               ASSET_PATH,
                               TILE_SIZE_M,
                               #fishnet, asset_df, 
                               OUTDIR,
                               ASSET_GDF_FN=None,
                               INPUT_NODATA_VALUE=0,
                               GRID_SIZE_M=30
                              ):
    '''
    A wrapper of a gently modified ee_download.download_image_by_asset_path that downloads a subtile (based on a vector 'fishnet') of a GEE asset tile.
    Stacks all bands into a multi-band geotiff
    
    SUBTILE_LOC : the index number of the subtile that, with the fishnet, will define the subtile region of the GEE asset tile in which to download the data
    # TILELOC     : the index of the GEE asset tile
    ID_NUM      : the id in the column name in the asset df indicating the GEE asset tile to import
    ID_COL      : the column used to identify the tile
    ASSET_PATH  : The GEE path to the image collection where the assets are stored
    TILE_SIZE_M : the dim in meters of 1 side of a subtile (too big, asset transfer fails; too small, too many subtiles are created)
    ASSET_GDF_FN: input asset gdf filename (this was uploaded to GEE and used to create the GEE aggregate tiles - instead of generating this dynamically from asset_df, which doesnt work nicely)
    OUTDIR      : the main ouput dir for this TILELOC into which many subdirs based on SUBTILE_LOC will be created
    INPUT_NODATA_VALUE : needs to be an int
    GRID_SIZE_M : the size in meters of the gridded data - used with TILE_SIZE_M to calc subtile length for fishnet
    
    ASSET_PATH gives you the following:
        fishnet : a vector fishnet based on the GEE asset tile bounds that defines the subtile regions
        asset_df : the asset data frame that bounds the tiles of the asset in this image collection
    '''
    
    fails = []
    
    # Use the GEE assett collection path (specifies the image collection in which the GEE asset tiles are stored) to make df and fishnet
    asset_df = get_gee_assets(ASSET_PATH)
    # Explode 'properties' field dict into multiple columns to get GEE metadata info nicely into columns of asset_df
    # this is needed to have ID_COL available in the data frame
    asset_df = pd.concat([asset_df.drop(['properties'], axis=1), asset_df['properties'].apply(pd.Series)], axis=1)
    
    # Replace this
    #ASSET_TILE_NAME = os.path.basename(asset_df.id.to_list()[TILELOC])
    ASSET_TILE_NAME = asset_df[asset_df[ID_COL] == ID_NUM]['system:index'].to_list()[0]
    
    # Make asset tile subdir
    OUTDIR_TILE = os.path.join(OUTDIR, ASSET_TILE_NAME)
    
    # if not os.path.exists(OUTDIR_TILE):
    #     os.makedirs(OUTDIR_TILE)
    
    # Hard-coded tile size
    # TODO: make into an argument for flexibility - too large and asset download will fail - too small and you have too many subtiles
    #TILE_SIZE_M = 500
    
    #print(f'\n\tSubtile {SUBTILE_LOC} for tile loc {TILELOC} to subdir: {OUTDIR_TILE}')
    print(f'\n\tSubtile {SUBTILE_LOC} for {ID_COL} # {ID_NUM} ({ASSET_TILE_NAME}) to subdir: {OUTDIR_TILE}')

    if False:
        fishnet_df = create_fishnet(asset_df[asset_df[ID_COL] == ID_NUM], TILE_SIZE_M)
        fishnet_df['subtile'] = fishnet_df.index
        fishnet_df['tile'] = ID_NUM
    else:
        # New fishnet approach does geo abyss (-180) crossing tiles (aka dateline tiles - a misnomer) correctly
        if ASSET_GDF_FN is None:
            # Get ASSET_GDF directly from GEE 
            TILE_NUM_LIST = asset_df[ID_COL].to_list()
            input_coordinates_list = [asset_df['system:footprint'][i]['coordinates'] for i, TILE_NUM in enumerate(TILE_NUM_LIST)]
            input_tiles_list = [TILE_NUM for i, TILE_NUM in enumerate(asset_df[ID_COL].to_list())]

            # Create a GeoDataFrame with the polygon
            asset_gdf = pd.concat([create_polygon_from_coordinates(input_coordinates) for input_coordinates in input_coordinates_list], ignore_index=True)
            asset_gdf = pd.concat([asset_gdf, pd.DataFrame({ID_COL: input_tiles_list})], axis=1)
        else:
            # Read GeoDataFrame directly
            asset_gdf = gpd.read_file(ASSET_GDF_FN)
            if not ID_COL in asset_gdf.columns:
                # Make an uppcase ID col thats the same as the lower case col - this works if input ID_COL is upper and gdf has lower
                asset_gdf[ID_COL] = asset_gdf[ID_COL.lower()]
        
        # Now get fishnet    
        fishnet_df = create_fishnet_new(asset_gdf[asset_gdf[ID_COL] == ID_NUM], TILE_SIZE_M * GRID_SIZE_M, ID_COL)
    
    try:
        
        # Change region CRS to that of the export tifs
        minx, miny, maxx, maxy = fishnet_df.to_crs(4326).iloc[SUBTILE_LOC].geometry.bounds
        region_4326 = ee.Geometry.BBox(minx, miny, maxx, maxy)
        
        # I thought this would bring the crs and the geometry needed 
        #region  = fishnet.iloc[SUBTILE_LOC]#.geometry

        # We want to submit to DPS this...
        #print('Fetching..')
        downloaded_image_fn = ee_download.download_image_by_asset_path(
                                asset_path = asset_df[asset_df[ID_COL] == ID_NUM]['id'].iloc[0], #asset_df.iloc[TILELOC]['id'],
                                output_folder = OUTDIR,
                                region = region_4326,
                                #crs = fishnet.crs,
                                idx = str(SUBTILE_LOC)
                                )
        #print(f'SUBTILE_LOC: {SUBTILE_LOC} : {downloaded_image_fn}')

        # Extract tif to zip
        out_subdir = os.path.splitext(downloaded_image_fn)[0]
        #print(f'out subdir: {out_subdir}')
        
        # Get the subtile string to add to each unzipped tif
        subtile_str = os.path.basename(os.path.splitext(downloaded_image_fn)[0]).split('-subtile')[1]
        #print(f'subtile str: {subtile_str}')
        
        with zipfile.ZipFile(downloaded_image_fn, 'r') as zip_ref:
            zip_ref.extractall(out_subdir)
            print(f'\tSUBTILE_LOC: {SUBTILE_LOC} : extracted tifs to {out_subdir}')
        os.remove(downloaded_image_fn)
        
        tif_fn_orig_list = glob.glob(out_subdir + '/*.tif')
        for tif_fn in tif_fn_orig_list:
            # Rename by appending subtile string
            tif_fn_new = os.path.splitext(tif_fn)[0] + f'-subtile{subtile_str}.tif'
            os.rename(tif_fn, tif_fn_new)
        
        tif_fn_new_list = glob.glob(out_subdir + '/*.tif') 
        print(tif_fn_orig_list)
        print(tif_fn_new_list)
        # Use rasterio to set band descriptions
        # Use part of orig filename as descriptions of tri-seasonal composites
        descriptions = [fn.split('.')[1].replace('.tif','') for fn in tif_fn_orig_list]  
        stack_tif_fn = os.path.join(out_subdir, os.path.basename(out_subdir) + '.tif')
        print(descriptions)
        #create_multiband_geotiff(stack_tif_fn, descriptions, tif_fn_new_list)
        create_multiband_cog(stack_tif_fn, descriptions, tif_fn_new_list, fishnet_df, input_nodata_value=INPUT_NODATA_VALUE)

        # Remove individual geotiffs
        for tif_fn in tif_fn_new_list:
            os.remove(tif_fn)
            
    except Exception as e:
        raise e
        fails.append(fishnet.iloc[SUBTILE_LOC])
        
    return fails
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtile_loc", type=int, required=True, help="The subtile index number provided by a fishnet over the GEE asset tile.")
    #parser.add_argument("--tile_loc", type=int, required=True, help="The tile index number of the GEE asset tile to import (based on df index (.iloc[x]) not tile id number)")
    parser.add_argument("--id_num", type=int, required=True, help="The id in the column name in the asset df indicating the GEE asset tile to import")
    parser.add_argument("--id_col", type=str, required=True, help="The column name in the asset df indicating the GEE asset tile numbers")
    parser.add_argument("--asset_path", type=str, required=True, default='projects/foreststructure/Circumboreal/S1_Composites_albers', help="The GEE path to the image collection where the assets are stored.")
    parser.add_argument("--asset_gdf_fn", type=str, required=True, default='https://maap-ops-workspace.s3.amazonaws.com/shared/montesano/databank/boreal_tiles_v004_agg12/boreal_tiles_v004_agg12.gpkg', help="GEE aggregate tiles")
    parser.add_argument("--tile_size_m", type=int, required=True, default=500, help="The dim in meters of 1 side of a subtile")
    parser.add_argument("--out_dir", type=str, required=True, help="The path where the subtile tifs will be imported")
    
    args = parser.parse_args()
    
    #fails = do_gee_download_by_subtile(args.subtile_loc, args.tile_loc, args.asset_path, args.out_dir)
    fails = do_gee_download_by_subtile(args.subtile_loc, args.id_num, args.id_col, args.asset_path, args.tile_size_m, args.out_dir, ASSET_GDF_FN=args.asset_gdf_fn)
    
    return fails

if __name__ == "__main__":
    main()
    
    