import json
import os
import rasterio as rio
import geopandas as gpd
from shapely.geometry import box
from fiona.crs import from_epsg
from rasterio.mask import mask
from rasterio.warp import *
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio import windows
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import rioxarray as rxr

import sys
# COG
import tarfile
import rasterio

from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rasterio.vrt import WarpedVRT

from rasterio.plot import show

from CovariateUtils import write_cog, get_index_tile

def GetBandLists(json_files, bandnum):
    BandList = []
    for j in json_files:
        inJSON = os.path.join(GeoJson_file,j)
        with open(inJSON) as f:
            response = json.load(f)
        for i in range(len(response['features'])):
            try:
                getBand = response['features'][i]['assets']['SR_B' + str(bandnum) + '.TIF']['href']
                BandList.append(getBand)
            except exception as e:
                print(e)
                
    BandList.sort()
    return BandList


parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=str, help="The filename of the stack's set of vector tiles")
    parser.add_argument("-n", "--tile_number", type=int, help="The id of a tile that will define the bounds of the raster stacking")
    parser.add_argument("-b", "--tile_buffer_m", type=int, default=None, help="The buffer size (m) applied to the extent of the specified stack tile")
    parser.add_argument("-l", "--infile_layer", type=str, defaut=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-p", "--data_path", type=str, help="The path to the S3 bucket")
    args = parser.parse_args()
    
    if args.stack_tile_fn == None:
        print("Input a filename of the vector tiles that represents the arrangement by which the output stacks will be organized")
        os._exit(1)
    elif args.stack_tile_id == None:
        print("Input a specific tile id from the vector tiles the organize the stacks")
        os._exit(1)

    geojson_path_albers = args.infile

    tile_n = args.tile_number

    # Get tile by number form GPKG. Store box and out crs
    tile_id = get_index_tile(geojson_path_albers, tile_n, layer = "boreal_tiles_albers")
    in_bbox = tile_id['bbox_4326']
    out_crs = tile_id['tile_crs']
    
    # Get path of Landsat data and organize them into lists by band number
    json_files = [file for file in os.listdir(args.data_path) if 'local' in file]
    BLUEBands = GetBandLists(json_files, 2)
    GREENBands = GetBandLists(json_files, 3)
    REDBands = GetBandLists(json_files, 4)
    NIRBands = GetBandLists(json_files, 5)
    SWIRBands = GetBandLists(json_files, 6)
    SWIR2Bands = GetBandLists(json_files,7)
    print(len(BLUEBands))


if __name__ == "__main__":
    main()
    
    

