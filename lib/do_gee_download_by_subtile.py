import os

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

def get_gee_assets(asset_path):
    # fetch EE assets
    COLL_ID = asset_path
    collection = ee.ImageCollection(COLL_ID)
    collection_items = collection.toList(1000).getInfo()
    # tranlsate collection to df
    collection_items_df = pd.DataFrame.from_dict(collection_items)

    return collection_items_df
def create_fishnet(collection_items_df, asset_num, dims, DEBUG=False):
    in_crs = 'BOUNDCRS[SOURCECRS[PROJCRS["unnamed",BASEGEOGCRS["GRS 1980(IUGG, 1980)",DATUM["unknown",ELLIPSOID["GRS80",6378137,298.257222101,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],CONVERSION["unnamed",METHOD["Albers Equal Area",ID["EPSG",9822]],PARAMETER["Latitude of 1st standard parallel",50,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",70,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Latitude of false origin",40,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of false origin",180,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8822]],PARAMETER["Easting at false origin",0,LENGTHUNIT["Meter",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["Meter",1],ID["EPSG",8827]]],CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["Meter",1]],AXIS["northing",north,ORDER[2],LENGTHUNIT["Meter",1]]]],TARGETCRS[GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]],ABRIDGEDTRANSFORMATION["Transformation from GRS 1980(IUGG, 1980) to WGS84",METHOD["Position Vector transformation (geog2D domain)",ID["EPSG",9606]],PARAMETER["X-axis translation",0,ID["EPSG",8605]],PARAMETER["Y-axis translation",0,ID["EPSG",8606]],PARAMETER["Z-axis translation",0,ID["EPSG",8607]],PARAMETER["X-axis rotation",0,ID["EPSG",8608]],PARAMETER["Y-axis rotation",0,ID["EPSG",8609]],PARAMETER["Z-axis rotation",0,ID["EPSG",8610]],PARAMETER["Scale difference",1,ID["EPSG",8611]]]]'
    
    try:
        asset = collection_items_df.iloc[asset_num]
        print('\tasset id =', asset['id'])

        rows = asset['bands'][0]['dimensions'][0]
        cols = asset['bands'][0]['dimensions'][1]
    
        print('\trows =', rows, 'cols = ', cols)
    
        minx = (asset['bands'][0]['crs_transform'][2])
        miny = (asset['bands'][0]['crs_transform'][5])
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
        #print("fishnet = ", fishnet.crs)
        ### TO DO ###
        # Write fishnet to disk
    except Exception as e:
        raise e
    
    return fishnet

# def create_fishnet_on_collection_polygon(collection_polygon, dims, DEBUG=False):
    
#     in_crs = 'BOUNDCRS[SOURCECRS[PROJCRS["unnamed",BASEGEOGCRS["GRS 1980(IUGG, 1980)",DATUM["unknown",ELLIPSOID["GRS80",6378137,298.257222101,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],CONVERSION["unnamed",METHOD["Albers Equal Area",ID["EPSG",9822]],PARAMETER["Latitude of 1st standard parallel",50,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",70,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Latitude of false origin",40,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of false origin",180,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8822]],PARAMETER["Easting at false origin",0,LENGTHUNIT["Meter",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["Meter",1],ID["EPSG",8827]]],CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["Meter",1]],AXIS["northing",north,ORDER[2],LENGTHUNIT["Meter",1]]]],TARGETCRS[GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]],ABRIDGEDTRANSFORMATION["Transformation from GRS 1980(IUGG, 1980) to WGS84",METHOD["Position Vector transformation (geog2D domain)",ID["EPSG",9606]],PARAMETER["X-axis translation",0,ID["EPSG",8605]],PARAMETER["Y-axis translation",0,ID["EPSG",8606]],PARAMETER["Z-axis translation",0,ID["EPSG",8607]],PARAMETER["X-axis rotation",0,ID["EPSG",8608]],PARAMETER["Y-axis rotation",0,ID["EPSG",8609]],PARAMETER["Z-axis rotation",0,ID["EPSG",8610]],PARAMETER["Scale difference",1,ID["EPSG",8611]]]]'
    
#     try:
#         #asset = collection_items_df.iloc[asset_num]
        
#         asset = collection_polygon
        
#         print('\tasset id =', asset['id'])

#         rows = asset['bands'][0]['dimensions'][0]
#         cols = asset['bands'][0]['dimensions'][1]
    
#         print('\trows =', rows, 'cols = ', cols)
    
#         minx = (asset['bands'][0]['crs_transform'][2])
#         miny = (asset['bands'][0]['crs_transform'][5])
#         maxx = minx + (cols*30)
#         maxy = miny - (rows*30)
    
#         if DEBUG:
#             print('\tbbox Albers=', minx, miny, maxx, maxy)

#         #create a fishnet: https://spatial-dev.guru/2022/05/22/create-fishnet-grid-using-geopandas-and-shapely/
#         gdf_tile = gpd.GeoDataFrame(index=[0],geometry=[Polygon([[minx,miny],[minx,maxy],[maxx,maxy],[maxx, miny]])],crs=in_crs)
    
#         # Get the extent of the shapefile
#         minX, minY, maxX, maxY = gdf_tile.total_bounds

#         # Create a fishnet
#         x, y = (minX, minY)
#         geom_array = []

#         # Polygon Size
#         square_size = dims*30
#         while y <= maxY:
#             while x <= maxX:
#                 geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
#                 geom_array.append(geom)
#                 x += square_size
#             x = minX
#             y += square_size

#         # Convert fishnet to gdf
#         fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs(in_crs)
#         #print("fishnet = ", fishnet.crs)
#         ### TO DO ###
#         # Write fishnet to disk
#     except Exception as e:
#         raise e
    
#     return fishnet
        
def do_gee_download_by_subtile(SUBTILE_LOC, TILELOC, ASSET_PATH, 
                               #fishnet, asset_df, 
                               OUTDIR):
    '''A wrapper of a gently modified ee_download.download_image_by_asset_path that downloads a subtile (based on a fishnet) of a GEE asset tile
    
    SUBTILE_LOC : the index number of the subtile that, with the fishnet, will define the subtile region of the GEE asset tile in which to download the data
    TILELOC : the index of the GEE asset tile
    ASSET_PATH : The GEE path to the image collection where the assets are stored
    OUTDIR : the main ouput dir for this TILELOC into which many subdirs based on SUBTILE_LOC will be created
    
    ASSET_PATH gives you the following:
        fishnet : a vector fishnet based on the GEE asset tile bounds that defines the subtile regions
        asset_df : the asset data frame that bounds the tiles of the asset in this image collection
    '''
    
    fails = []
    
    # Use the GEE assett collection path (specifies the image collection in which the GEE asset tiles are stored) to make df and fishnet
    asset_df = get_gee_assets(ASSET_PATH)
    ASSET_TILE_NAME = os.path.basename(asset_df.id.to_list()[TILELOC])
    
    # Make asset tile subdir
    OUTDIR_TILE = os.path.join(OUTDIR, ASSET_TILE_NAME)
    
    # if not os.path.exists(OUTDIR_TILE):
    #     os.makedirs(OUTDIR_TILE)
    
    print(f'\n\tSubtile {SUBTILE_LOC} for tile loc {TILELOC} to subdir: {OUTDIR_TILE}')
    fishnet_df = create_fishnet(asset_df, TILELOC, 500)
    fishnet_df['subtile'] = fishnet_df.index
    fishnet_df['tile'] = TILELOC
    
    try:
        
        # Change region CRS to that of the export tifs
        minx, miny, maxx, maxy = fishnet_df.to_crs(4326).iloc[SUBTILE_LOC].geometry.bounds
        region_4326 = ee.Geometry.BBox(minx, miny, maxx, maxy)
        
        # I thought this would bring the crs and the geometry needed 
        #region  = fishnet.iloc[SUBTILE_LOC]#.geometry

        # We want to submit to DPS this...
        #print('Fetching..')
        downloaded_image_fn = ee_download.download_image_by_asset_path(
                                asset_path = asset_df.iloc[TILELOC]['id'],
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
        
        for tif_fn in glob.glob(out_subdir + '/*.tif'):
            # Rename by appending subtile string
            tif_fn_new = os.path.splitext(tif_fn)[0] + f'-subtile{subtile_str}.tif'
            os.rename(tif_fn, tif_fn_new)
            
    except Exception as e:
        raise e
        fails.append(fishnet.iloc[SUBTILE_LOC])
        
    return fails
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtile_loc", type=int, required=True, help="The subtile index number provided by a fishnet over the GEE asset tile.")
    parser.add_argument("--tile_loc", type=int, required=True, help="The tile index number of the GEE asset tile to import (based on df index (.iloc[x]) not tile id number)")
    parser.add_argument("--asset_path", type=str, required=True, default='projects/foreststructure/Circumboreal/S1_Composites_albers', help="The GEE path to the image collection where the assets are stored.")
    #parser.add_argument("--fishnet_df", required=True, help="The data frame of the fishnet in 4326")
    #parser.add_argument("--asset_df", required=True, help="The GEE asset data frame")
    parser.add_argument("--out_dir", type=str, required=True, help="The path where the subtile tifs will be imported")
    
    args = parser.parse_args()
    
    #fails = do_gee_download_by_subtile(args.subtile_loc, args.tile_loc, args.fishnet_df, args.asset_df, args.out_dir)
    fails = do_gee_download_by_subtile(args.subtile_loc, args.tile_loc, args.asset_path, args.out_dir)
    
    return fails

if __name__ == "__main__":
    main()
    
    