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