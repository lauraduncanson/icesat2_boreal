import numpy
from rasterio import enums
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.session import AWSSession
from rasterio.warp import array_bounds, calculate_default_transform
from rio_cogeo.profiles import cog_profiles
from rio_tiler.utils import create_cutline
from rio_cogeo.cogeo import cog_translate
from rio_tiler.models import ImageData
from rio_tiler.mosaic import mosaic_reader
from rio_tiler.io import COGReader
import rasterio
import rasterio.mask
import geopandas
import os
import boto3
from cachetools import FIFOCache, TLRUCache, cached
from datetime import datetime, timedelta, timezone
from typing import List
import pandas as pd

try:
    from maap.maap import MAAP
    # create MAAP class
    #maap = MAAP(maap_host='api.maap-project.org')
    maap = MAAP()
    HAS_MAAP = True
except ImportError:
    print('NASA MAAP is unavailable')
    HAS_MAAP = False
    
# Set up the endpoints for the aws s3 credentials for various DAACs    
s3_cred_endpoint_DAAC = {
    'podaac':   'https://archive.podaac.earthdata.nasa.gov/s3credentials',
    'gesdisc':  'https://data.gesdisc.earthdata.nasa.gov/s3credentials',
    'lpdaac':   'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials',
    'ornldaac': 'https://data.ornldaac.earthdata.nasa.gov/s3credentials',
    'ghrcdaac': 'https://data.ghrc.earthdata.nasa.gov/s3credentials'
}

def creds_expiration_timestamp(_key, creds, _now) -> float:
    """Return the expiration time of an AWS credentials object converted to the
    number of seconds (fractional) from the epoch in UTC, minus 5 minutes."""

    # Credentials from `get_creds` contains the key "Expiration" with a `datetime`
    # value, while credentials from `get_creds_DAAC` contains the key "expiration"
    # with a `str` value.

    expiration = creds.get("Expiration") or creds["expiration"]
    expiration_dt = (
        expiration
        if isinstance(expiration, datetime)  # from AWS STS (ref: boto3 docs)
        else datetime.strptime(expiration, "%Y-%m-%d %H:%M:%S%z") # from MAAP
    )
    expiration_dt_utc = expiration_dt.astimezone(timezone.utc)

    # Subtract 5 minutes for "wiggle room"
    return (expiration_dt_utc - timedelta(minutes=5)).timestamp()

def get_index_tile(vector_path: str, id_col: str, tile_id: int, buffer: float = 0, layer: str = None):
    '''
    Given a vector tile index, select by id the polygon and return
    GPKG is the recommended vector format - single file, includes projection, can contain multiple variants and additional information.
    TODO: should it be a class or dict
    
    
    vector_path: str
        Path to GPKG file
    buffer: float
        Distance to buffer geometry in units of layer
    id_col: str
        Column name of the tile_id
    tile_id: int
        Tile ID to extract/build info for
        
    returns:
        geopandas.geodataframe.GeoDataFrame,
            Polygon in original crs
        geopandas.geoseries.GeoSeries,
            Polygon of Buffered in original crs
        list,
            Bounds of original polygon
        rasterio.crs.CRS,
            Coordinate Reference System of original tile
        geopandas.geodataframe.GeoDataFrame,
            4326 Polygon
        list,
            Bounds in 4326
        geopandas.geoseries.GeoSeries,
            Polygon of Buffered in 4326
        list
            Buffered Bounds in 4326
    Usage:
    get_index_tile(
        vector_path = '/projects/maap-users/alexdevseed/boreal_tiles.gpkg',
        tile_id = 30542,
        buffer = 120
        )
    
    '''
    
    tile_parts = {}

    if layer is None:
        layer = os.path.splitext(os.path.basename(vector_path))[0]
        
    if '.parquet' in vector_path:
        tile_index = geopandas.read_parquet(vector_path)
    else:
        tile_index = geopandas.read_file(vector_path, layer=layer)

    tile_parts["geom_orig"] = tile_index[tile_index[id_col]==tile_id]
    tile_parts["geom_orig_buffered"] = tile_parts["geom_orig"]["geometry"].buffer(buffer)
    tile_parts["bbox_orig"] = tile_parts["geom_orig"].bounds.iloc[0].to_list()
    tile_parts["tile_crs"] = CRS.from_wkt(tile_index.crs.to_wkt()) #A rasterio CRS object

    # Properties of 4326 version of tile
    tile_parts["geom_4326"] = tile_parts["geom_orig"].to_crs(4326)
    tile_parts["bbox_4326"] = tile_parts["geom_4326"].bounds.iloc[0].to_list()
    tile_parts["geom_4326_buffered"] =  tile_parts["geom_orig_buffered"].to_crs(4326)
    tile_parts["bbox_4326_buffered"] = tile_parts["geom_4326_buffered"].bounds.iloc[0].to_list()
    
    return tile_parts

def reader(src_path: str, bbox: List[float], epsg: CRS, dst_crs: CRS, height: int, width: int, bands: tuple) -> ImageData:
    with COGReader(src_path) as cog:
        return cog.part(bbox, bounds_crs=epsg, max_size=None, dst_crs=dst_crs, height=height, width=width, indexes=bands)

def get_shape(bbox, res=30):
    left, bottom, right, top = bbox
    width = int((right-left)/res)
    height = int((top-bottom)/res)
    return height,width

def write_cog(stack, out_fn: str, in_crs, src_transform, bandnames: list, out_crs=None, resolution: tuple=(30, 30), clip_geom=None, clip_crs=None, align:bool=False, input_nodata_value:int=None, resampling='nearest'):
    '''
    Write a cloud optimized geotiff with compression from a numpy stack of bands with labels
    Reproject if needed, Clip to bounding box if needed.
    
    Parameters:
    stack: np.array 
        3d numpy array (bands, height, width) 
    out_fn: str
        Output Filename
    in_crs: str
        CRS of input raster
    src_transform: Affine
        Affine transform of input raster
    bandnames: list[str]
        List of bandnames in band/dimension order that matches stack
    out_crs: CRS, optional
        If reprojecting, the output CRS
    clip_geom: dict, optional
        Polygon geometry as geojson dictionary
    clip_crs: CRS, optional
        CRS of clip_geom
    align: bool, optional
        True aligns the output raster with the top left corner of the clip_geom. clip_geom CRS must be the same as
        the out_crs.
    input_nodata_value: int, optional
        Setting to an int allows for reading in uint8 datatypes; otherwise assume np.nan
    '''
    
    #TODO: remove print statements, add debugging
    
    print('Shape of input:\t\t\t',stack.shape)
    
    if out_crs is None:
        out_crs = in_crs
        
    if input_nodata_value is None:
        print('Input nodata isnt provided; assuming NaN...')
        input_nodata_value = numpy.nan
   
    # Set the profile for the in memory raster based on the ndarry stack
    src_profile = dict(
        driver="GTiff",
        height=stack.shape[1],
        width=stack.shape[2],
        count=stack.shape[0],
        dtype=stack.dtype,
        crs=in_crs,
        transform=src_transform,
        nodata=input_nodata_value)

    # Set the reproject parameters for the WarpedVRT read
    vrt_params = {}
    if out_crs is not None:
        vrt_params["crs"] = out_crs
        vrt_params["src_crs"] = in_crs
        vrt_params["dtype"] = str(stack.dtype)
        vrt_params["nodata"] = input_nodata_value
        vrt_params["resampling"] = enums.Resampling.nearest # nearest prevents contamination from nan values and maintains numerical categories
        if resampling == 'cubic': vrt_params["resampling"] = enums.Resampling.cubic # needed for topo
        if resampling == 'bilinear': vrt_params["resampling"] = enums.Resampling.bilinear # needed for topo
        
        print(f"Resampling:\t\t\t {vrt_params['resampling']}\t[0=nearest, 1=blinear, 2=cubic]")
        
        #TODO: Add  transform with resolution specification
        if out_crs != in_crs:
            left, bottom, right, top = array_bounds(height = src_profile["height"],
                   width = src_profile["width"], 
                   transform = src_profile["transform"])
            #print('\n\n\n\nDEBUG----------\n\n\n\n')
            #print(f"{left}, {bottom}, {right}, {top}")            
            vrt_params["transform"], vrt_params["width"], vrt_params["height"] = calculate_default_transform(
                src_crs = in_crs,
                dst_crs = out_crs,
                left = left,
                bottom = bottom,
                right = right,
                top = top,
                width = src_profile["width"],
                height = src_profile["height"],
                resolution = resolution
                )
            print(f"Shape after transform:\t\t({src_profile['count']},{vrt_params['width']},{vrt_params['height']})")
        if align is True:
            # TODO: here, clip_geom will only work as a cutline *if* [1] its a box and [2] it's in its original prj
            left, bottom, right, top = clip_geom.total_bounds # tot bounds only works if this is in the final grid you want - but its not - so a cutline in the FINAL crs is still needed
            vrt_params["transform"], vrt_params["width"], vrt_params["height"] = calculate_default_transform(
                src_crs = in_crs,
                dst_crs = out_crs,
                left = left,
                bottom = bottom,
                right = right,
                top = top,
                width = src_profile["width"],
                height = src_profile["height"],
                resolution = resolution
                )
            print(f"Shape after clip & transform:\t ({src_profile['count']},{vrt_params['width']},{vrt_params['height']})")
            
    print('Output resolution:\t\t',resolution)
        
    # Get the rio-cogeo profile for deflate compression, modify some of the options
    dst_profile = cog_profiles.get("deflate")
    dst_profile['blockxsize']=256
    dst_profile['blockysize']=256
    dst_profile['predictor']=1 # originally set to 2, which fails with 'int'; 1 tested successfully for 'int' and 'float64'
    dst_profile['zlevel']=7
    dst_profile["nodata"] = input_nodata_value # Updated to get nodata value into output after turning off clip_geom functionality
    
    with MemoryFile() as memfile:
        with memfile.open(**src_profile) as mem:
            # Populate the input file with NumPy array
            # HERE; this memory file can be reprojected then saved
            mem.write(stack)

            if clip_geom is not None:
                if False:
                    print('\tDEBUGFix clip attempt #1: try rasterio.mask.mask')
                    # https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
                    # Try rasterio mask
                    clip_geom_json = clip_geom.__geo_interface__['features'][0]['geometry']
                    clipped_stack, out_transform = rasterio.mask.mask(mem, [clip_geom_json], crop=True, nodata=input_nodata_value)
                    vrt_params.update({
                                #"driver": "GTiff",
                                 "height": vrt.shape[1],
                                 "width": vrt.shape[2],
                                 "transform": out_transform})
                    print(f"Current stack shape:\t\t({mem.profile['count']},{vrt_params['width']},{vrt_params['height']})")
                
                # NOTE: This is now OFF - which fixes the jagged effect of nodata at clipline edges!!
                if False:
                    # TODO: Here is what topo needs to get the clip correct - but not working as expected yet
                    if out_crs != in_crs:
                        print('Reprojecting clip geom ...')
                        clip_geom = clip_geom.to_crs(out_crs)
                        clip_crs = out_crs
                        # Do the clip to geometry (rasterio takes this; not in_bbox)
                        # # https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
                        print(f"Clipping (in memory) with geom...")
                        print("\tDEBUG: TODO - is this actually working?")
                        print('\tDEBUG: Fix clip attempt #2: get cutline working')
                        clip_geom_json = clip_geom.__geo_interface__['features'][0]['geometry']
                        vrt_params["cutline"] = create_cutline(mem, clip_geom_json, geometry_crs = clip_crs)
                             
            print('Writing img to memory...')
            
            for n in range(len(bandnames)):
                mem.set_band_description(n+1, bandnames[n])
        
            with WarpedVRT(mem,  **vrt_params) as vrt:
                print(f"Current stack shape:\t\t({vrt.profile['count']},{vrt.profile['width']},{vrt.profile['height']})")                
                cog_translate(
                    vrt,
                    # To avoid rewriting over the infile
                    out_fn,
                    dst_profile,
                    add_mask=True,
                    in_memory=True,
                    quiet=False)

    print('Image written to disk:\t\t', out_fn)
    # TODO: return something useful
    return True

@cached(
    cache=TLRUCache(
        maxsize=1,
        ttu=creds_expiration_timestamp,
        timer=lambda: datetime.now(timezone.utc).timestamp(),
    )
)

def get_creds():
    """Get temporary credentials by assuming role"""
    sts_client = boto3.client('sts')
    assumed_role_object=sts_client.assume_role(
        RoleArn="arn:aws:iam::884094767067:role/maap-bucket-access-role", 
        RoleSessionName="AssumeRoleSession1"
    )
    return assumed_role_object['Credentials']

@cached(
    cache=TLRUCache(
        maxsize=1,
        ttu=creds_expiration_timestamp,
        timer=lambda: datetime.now(timezone.utc).timestamp(),
    )
)
def get_creds_DAAC():
    return maap.aws.earthdata_s3_credentials(
        'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials'
    )

@cached(cache=FIFOCache(maxsize=1), key=lambda creds: creds["SessionToken"])
def get_aws_session(creds):
    """Create a Rasterio AWS Session with Credentials"""
    #creds = get_creds()
    boto3_session = boto3.Session(
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken']
    )
    return AWSSession(boto3_session, requester_pays=True)

@cached(cache=FIFOCache(maxsize=1), key=lambda creds: creds["sessionToken"])
def get_aws_session_DAAC(creds):
    """Create a Rasterio AWS Session with Credentials"""
    #creds = maap.aws.earthdata_s3_credentials('https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials')
    boto3_session = boto3.Session(
        aws_access_key_id=creds['accessKeyId'], 
        aws_secret_access_key=creds['secretAccessKey'],
        aws_session_token=creds['sessionToken'],
        region_name='us-west-2'
    )
    return AWSSession(boto3_session)

#Return a common mask for a set of input ma
def common_mask(ma_list, apply=False):
    if type(ma_list) is not list:
        print("Input must be list of masked arrays")
        return None
    #Note: a.mask will return single False if all elements are False
    #numpy.ma.getmaskarray(a) will return full array of False
    #ma_list = [numpy.ma.array(a, mask=numpy.ma.getmaskarray(a), shrink=False) for a in ma_list]
    a = numpy.ma.array(ma_list, shrink=False)
    #Check array dimensions
    #Check dtype = bool
    #Masked values are listed as true, so want to return any()
    #a+b+c - OR (any)
    mask = numpy.ma.getmaskarray(a).any(axis=0)
    #a*b*c - AND (all)
    #return a.all(axis=0)
    if apply:
        return [numpy.ma.array(b, mask=mask) for b in ma_list] 
    else:
        return mask
    
def parse_aws_creds(credentials_fn):
    
    import configparser
    
    config = configparser.ConfigParser()
    config.read(credentials_fn)
    profile_name = config.sections()[0]
    
    aws_access_key_id = config['boreal_pub']['aws_access_key_id']
    aws_secret_access_key = config['boreal_pub']['aws_secret_access_key']
    
    return profile_name, aws_access_key_id, aws_secret_access_key

def get_s3_fs_from_creds(credentials_fn=None, anon=False):
    
    if anon:
        s3_fs = s3fs.S3FileSystem(anon=anon)
    else:
        if credentials_fn is not None:
            profile_name, aws_access_key_id, aws_secret_access_key = parse_aws_creds(credentials_fn)
            s3_fs = s3fs.S3FileSystem(profile=profile_name, key=aws_access_key_id, secret=aws_secret_access_key)
        else:
            print("Must provide a credentials filename if anon=False")
            os._exit(1)
    return s3_fs

def get_rio_aws_session_from_creds(credentials_fn):
    
    import s3fs
    import rasterio as rio
    from rasterio.session import AWSSession
    import boto3
    import pandas as pd
    
    profile_name, aws_access_key_id, aws_secret_access_key = parse_aws_creds(credentials_fn)

    boto3_session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            #aws_session_token=credentials['SessionToken'],
            profile_name=profile_name
        )

    rio_aws_session = AWSSession(boto3_session)
    
    return rio_aws_session

def local_to_s3(url, user='montesano'):
    ''' A Function to convert local paths to s3 urls'''
    return url.replace('/projects/my-private-bucket', f's3://maap-ops-workspace/{user}')

def get_stack_fn(stack_list_fn, in_tile_num, user, col_name='local_path', return_s3=True):
    
    '''Return the path of the raster stack for in_tile_num from a list of stack paths (stack_list_fn) from *tindex_master.csv
    '''
    all_stacks_df = pd.read_csv(stack_list_fn)
    
    # Get the s3 location from the location (local_path) indicated in the tindex master csv
    if user is None:
        # Get user specific to each file in row of tindex table: this is safer, in case multiple tables from different users have been concatenated 
        # user is at position 3 of s3_path
        USER_POS = 3
        all_stacks_df['user'] = all_stacks_df['s3_path'].str.split('/', expand=True)[USER_POS]
        all_stacks_df['s3'] = [local_to_s3(local_path, all_stacks_df.user.to_list()[i]) for i, local_path in enumerate(all_stacks_df[col_name]) ]
    else:
        all_stacks_df['s3'] = [local_to_s3(local_path, user) for local_path in all_stacks_df[col_name]]
    
    if return_s3:
        col_name = 's3'
    
    stack_for_tile = all_stacks_df[all_stacks_df[col_name].str.contains("_"+str(in_tile_num)+"_")]
    
    print("\nGetting stack fn from: ", stack_list_fn)
    [print(i) for i in stack_for_tile[col_name].to_list()]
    stack_for_tile_fn = stack_for_tile[col_name].to_list()[0]
    
    if len(stack_for_tile)==0:
        stack_for_tile_fn = None
        
    return stack_for_tile_fn