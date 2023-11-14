import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ee
from shapely.geometry import Polygon
import contextily as ctx
from shapely import geometry
import ee_download
import argparse


def get_gee_assets(asset_path):
    # fetch EE assets
    COLL_ID = asset_path
    collection = ee.ImageCollection(COLL_ID)
    collection_items = collection.toList(1000).getInfo()
    # tranlsate collection to df
    collection_items_df = pd.DataFrame.from_dict(collection_items)

    return collection_items_df

def create_fishnet(collection_items_df, asset_num, dims):
    in_crs = 'BOUNDCRS[SOURCECRS[PROJCRS["unnamed",BASEGEOGCRS["GRS 1980(IUGG, 1980)",DATUM["unknown",ELLIPSOID["GRS80",6378137,298.257222101,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],CONVERSION["unnamed",METHOD["Albers Equal Area",ID["EPSG",9822]],PARAMETER["Latitude of 1st standard parallel",50,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",70,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Latitude of false origin",40,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8821]],PARAMETER["Longitude of false origin",180,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8822]],PARAMETER["Easting at false origin",0,LENGTHUNIT["Meter",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["Meter",1],ID["EPSG",8827]]],CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["Meter",1]],AXIS["northing",north,ORDER[2],LENGTHUNIT["Meter",1]]]],TARGETCRS[GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]],ABRIDGEDTRANSFORMATION["Transformation from GRS 1980(IUGG, 1980) to WGS84",METHOD["Position Vector transformation (geog2D domain)",ID["EPSG",9606]],PARAMETER["X-axis translation",0,ID["EPSG",8605]],PARAMETER["Y-axis translation",0,ID["EPSG",8606]],PARAMETER["Z-axis translation",0,ID["EPSG",8607]],PARAMETER["X-axis rotation",0,ID["EPSG",8608]],PARAMETER["Y-axis rotation",0,ID["EPSG",8609]],PARAMETER["Z-axis rotation",0,ID["EPSG",8610]],PARAMETER["Scale difference",1,ID["EPSG",8611]]]]'
    
    try:
        asset = collection_items_df.iloc[asset_num]
        print('asset id =', asset['id'])

        rows = asset['bands'][0]['dimensions'][0]
        cols = asset['bands'][0]['dimensions'][1]
    
        print('rows =', rows)
        print('cols = ', cols)
    
        minx = (asset['bands'][0]['crs_transform'][2])
        miny = (asset['bands'][0]['crs_transform'][5])
        maxx = minx + (cols*30)
        maxy = miny - (rows*30)
    
        print('bbox Albers=', minx, miny, maxx, maxy)

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
        print("fishnet = ", fishnet.crs)
        ### TO DO ###
        # Write fishnet to disk
    except Exception as e:
        raise e
    
    return fishnet

def download_to_maap(fishnet, outdir, collection_items_df, tile_num):
    fails = []
    # iterate over fishnet and import GEE image
    fishnet_4326 = fishnet.to_crs("EPSG:4326")
    for index, row in fishnet_4326.iterrows():
        try:
            print(index)
            minx, miny, maxx, maxy = fishnet_4326.iloc[index].geometry.bounds
            print(minx, miny, maxx, maxy)
            region = ee.Geometry.BBox(minx, miny, maxx, maxy)

            print('Fetching..')
            downloaded_image = ee_download.download_image_by_asset_path(
            asset_path = collection_items_df.iloc[tile_num]['id'],
            output_folder = outdir,
            region = region,
            idx = str(index)
            )
            print('Done')
        except Exception as e:
            raise e
            #fails.append(fishnet.iloc[index])
            
            
    return fails

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--tile_num", type=int, required=True, help="The number of the S1 tile to import (based on df index (.iloc[x]) not tile id number)")
    parser.add_argument("-d", "--dims", type=int, required=True, default=500, help="The dimensions of the fishnet in pixels")
    parser.add_argument("-i", "--asset_path", type=str, required=True, default="projects/foreststructure/Circumboreal/S1_Composites_albers", help="The GEE path to the asset collection")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="The path where the files will be imported to")
    
    args = parser.parse_args()
    
    asset_df = get_gee_assets(args.asset_path)
    
    fishnet_df = create_fishnet(asset_df, args.tile_num, args.dims)
    
    fails = download_to_maap(fishnet_df, args.out_dir, asset_df, args.tile_num)
    
    #print(fails)
    
if __name__ == "__main__":
    main()