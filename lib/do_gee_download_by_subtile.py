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

#def create_fishnet(collection_items_df, asset_num, dims, DEBUG=False):
def create_fishnet(asset, dims, DEBUG=False):
    
    in_crs = 'BOUNDCRS[SOURCECRS[PROJCRS["unnamed",BASEGEOGCRS["GRS 1980(IUGG,\
    1980)",DATUM["unknown",ELLIPSOID["GRS80",6378137,298.257222101,\
    LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],CONVERSION["unnamed",METHOD["Albers Equal Area",ID["EPSG",9822]],\
    PARAMETER["Latitude of 1st standard parallel",50,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],\
    PARAMETER["Latitude of 2nd standard parallel",70,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Latitude of false origin",40,\
    ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of false origin",180,ANGLEUNIT["degree",0.0174532925199433],\
    ID["EPSG",8822]],PARAMETER["Easting at false origin",0,LENGTHUNIT["Meter",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["Meter",1],ID["EPSG",8827]]],CS[Cartesian,2],\
    AXIS["easting",east,ORDER[1],LENGTHUNIT["Meter",1]],AXIS["northing",north,ORDER[2],LENGTHUNIT["Meter",1]]]],\
    TARGETCRS[GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],\
    PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1]\
    ,ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],\
    ID["EPSG",4326]]],ABRIDGEDTRANSFORMATION["Transformation from GRS 1980(IUGG, 1980) to WGS84",METHOD["Position Vector transformation (geog2D domain)",ID["EPSG",9606]],\
    PARAMETER["X-axis translation",0,ID["EPSG",8605]],PARAMETER["Y-axis translation",0,ID["EPSG",8606]],PARAMETER["Z-axis translation",0,ID["EPSG",8607]],\
    PARAMETER["X-axis rotation",0,ID["EPSG",8608]],PARAMETER["Y-axis rotation",0,ID["EPSG",8609]],PARAMETER["Z-axis rotation",0,ID["EPSG",8610]],PARAMETER["Scale difference",1,ID["EPSG",8611]]]]'
    
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
    
        #minx = (asset['bands'][0][0]['crs_transform'][2])
        #miny = (asset['bands'][0][0]['crs_transform'][5])
        minx = asset['crs_transform'].iloc[0][2]
        miny = asset['crs_transform'].iloc[0][5]
        maxx = minx + (cols*30)
        maxy = miny - (rows*30)
    
        if DEBUG:
            print('\tbbox Albers=', minx, miny, maxx, maxy)

        #create a fishnet: https://spatial-dev.guru/2022/05/22/create-fishnet-grid-using-geopandas-and-shapely/
        gdf_tile = gpd.GeoDataFrame(index=[0],geometry=[Polygon([[minx,miny],[minx,maxy],[maxx,maxy],[maxx, miny]])],crs=in_crs)
    
        # Get the extent of the shapefile
        minX, minY, maxX, maxY = gdf_tile.total_bounds

        # Create a fishnet
        x, y = (minX, minY)
        geom_array = []

        # Polygon Size
        square_size = dims*30
        while y <= maxY:
            while x <= maxX:
                geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
                geom_array.append(geom)
                x += square_size
            x = minX
            y += square_size

        # Convert fishnet to gdf
        fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs(in_crs)

    except Exception as e:
        raise e
    
    return fishnet

import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
import numpy as np
import rasterio

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
                # if src_band.dtype != profile['dtype']:
                #     raise ValueError(f"Data type of {single_band_geotiffs[i-1]} does not match the expected data type.")
                # if src_band.crs != profile['crs']:
                #     raise ValueError(f"Projection of {single_band_geotiffs[i-1]} does not match the expected projection.")

                # Read and write each band to the new multiband geotiff
                dst.write(src_band.read(1), indexes=i)
                dst.set_band_description(i, band_name)

    print(f"Multiband geotiff created successfully at: {output_path}")

# TODO: use an id field 'AGG_TILE_NUM'
# replace TILE_LOC with id number of id field
def do_gee_download_by_subtile(SUBTILE_LOC, 
                               #TILELOC,
                               ID_NUM,
                               ID_COL,
                               ASSET_PATH,
                               TILE_SIZE_M,
                               #fishnet, asset_df, 
                               OUTDIR):
    '''
    A wrapper of a gently modified ee_download.download_image_by_asset_path that downloads a subtile (based on a vector 'fishnet') of a GEE asset tile.
    Stacks all bands into a multi-band geotiff
    
    SUBTILE_LOC : the index number of the subtile that, with the fishnet, will define the subtile region of the GEE asset tile in which to download the data
    # TILELOC     : the index of the GEE asset tile
    ID_NUM      : the id in the column name in the asset df indicating the GEE asset tile to import
    ID_COL      : the column used to identify the tile
    ASSET_PATH  : The GEE path to the image collection where the assets are stored
    TILE_SIZE_M : the dim in meters of 1 side of a subtile (too big, asset transfer fails; too small, too many subtiles are created)
    OUTDIR      : the main ouput dir for this TILELOC into which many subdirs based on SUBTILE_LOC will be created
    
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
    #fishnet_df = create_fishnet(asset_df, TILELOC, TILE_SIZE_M)
    fishnet_df = create_fishnet(asset_df[asset_df[ID_COL] == ID_NUM], TILE_SIZE_M)
    fishnet_df['subtile'] = fishnet_df.index
    fishnet_df['tile'] = ID_NUM
    
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
        # Use rasterio to set band descriptions
        # Use part of orig filename as descriptions of tri-seasonal composites
        descriptions = [fn.split('.')[1].replace('.tif','') for fn in tif_fn_orig_list]  
        stack_tif_fn = os.path.join(out_subdir, os.path.basename(out_subdir) + '.tif')

        create_multiband_geotiff(stack_tif_fn, descriptions, tif_fn_new_list)
    
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
    parser.add_argument("--tile_size_m", type=int, required=True, default=500, help="The dim in meters of 1 side of a subtile")
    parser.add_argument("--out_dir", type=str, required=True, help="The path where the subtile tifs will be imported")
    
    args = parser.parse_args()
    
    #fails = do_gee_download_by_subtile(args.subtile_loc, args.tile_loc, args.asset_path, args.out_dir)
    fails = do_gee_download_by_subtile(args.subtile_loc, args.id_num, args.id_col, args.asset_path, args.tile_size_m, args.out_dir)
    
    return fails

if __name__ == "__main__":
    main()
    
    