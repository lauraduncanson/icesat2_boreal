import argparse
import os
import geopandas as gpd
import boto3
import rasterio as rio
import geopandas as gpd
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.warp import *
from rasterio.merge import merge
from rasterio.crs import CRS
from rio_tiler.io import COGReader
import numpy as np
from rasterio.session import AWSSession
from typing import List
from CovariateUtils import write_cog, get_index_tile, get_aws_session, get_aws_session_DAAC, common_mask, get_shape, reader
#from fetch_HLS import get_HLS_data, get_LC2SR_data, get_ms_data, MS_BANDS_DICT
from fetch_from_api import get_ms_data, MS_BANDS_DICT
import json
import datetime
from CovariateUtils import get_creds, get_creds_DAAC
import multiprocessing as mp

from maap.maap import MAAP
maap = MAAP()

import numpy as np
from sklearn.cluster import MeanShift

def print_array_stats(result):
    print(f"\tPrinting array stats: {result.shape}")
    # Count of valid (non-NaN) pixels
    valid_pixel_count = np.count_nonzero(~np.isnan(result))
    print(f"\t\tValid Pixel Count: {valid_pixel_count}")
    print(f"\t\tMean: {np.nanmean(result):.2f}")
    print(f"\t\tStandard Deviation: {np.nanstd(result):.2f}")
    print(f"\t\tMinimum: {np.nanmin(result):.2f}")
    print(f"\t\tMaximum: {np.nanmax(result):.2f}")
    if not np.any(np.isnan(result)):
        # print(np.sum(~np.isnan(result)), result[~np.isnan(result)].flatten().shape())
        print(f"\t\t25th Percentile: {np.nanpercentile(np.ma.filled(result, np.nan), 25):.2f}")
        print(f"\t\t50th Percentile (Median): {np.nanpercentile(np.ma.filled(result, np.nan), 50):.2f}")
        print(f"\t\t75th Percentile: {np.nanpercentile(np.ma.filled(result, np.nan), 75):.2f}")

def nanpercentile_index_chunk(chunk, percentile, axis):
    """
    Calculate the indices of a given percentile in a chunk of a 3D numpy array.
    
    Parameters:
    - chunk (np.array): A chunk of the original 3D numpy array.
    - percentile (float): The percentile to compute (0-100).
    - axis (int): The axis along which to calculate percentiles (must match with context).
    
    Returns:
    - Indices array for the chunk.
    """
    return nanpercentile_index(chunk, percentile, axis)

def partition_array(arr, num_chunks, axis=0):
    """
    Partition a 3D array into chunks along the specified axis.
    
    Parameters:
    - arr (np.array): 3D numpy array to partition.
    - num_chunks (int): Number of chunks to create.
    - axis (int): The axis along which to create chunks.
    
    Returns:
    - List of numpy array chunks.
    """
    chunk_sizes = np.array_split(np.arange(arr.shape[axis]), num_chunks)
    return [np.take(arr, indices, axis=axis) for indices in chunk_sizes]

def multiprocess_nanpercentile_index(arr, percentile, axis=0, num_processes=4):
    """
    Multiprocess the nanpercentile_index to calculate the index of a percentile value.
    
    Parameters:
    - arr (np.array): A 3D numpy array.
    - percentile (float): The percentile to compute (0-100).
    - axis (int): Axis to compute the percentile along.
    - num_processes (int): Number of processes for multiprocessing.
    
    Returns:
    - Full index array similar in dimensions to input array.
    """
    
    # Divide the array into chunks
    chunks = partition_array(arr, num_processes, axis=axis)
    
    # Run computation in parallel
    pool = mp.Pool(processes=num_processes)
    results = pool.starmap(nanpercentile_index_chunk, [(chunk, percentile, axis) for chunk in chunks])
    
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
    
    # Combine the results back into a complete array
    index_array = np.concatenate(results, axis=axis)
    
    return index_array

def nanpercentile_index(arr, percentile, axis=0, DO_DASK=True, no_data_value=-9999):
    """
    Calculate the indices of a given percentile in a 3D numpy array while ignoring NaNs.

    Parameters:
    - arr (np.array): A 3D numpy array.
    - percentile (float): The percentile to compute (0-100).
    - axis (int): The axis along which to calculate the percentiles.

    Returns:
    - index_array: Indices of the calculated percentile values along the specified axis.
    """
    all_nan_mask = np.all(np.isnan(arr), axis=axis)
    arr[0, all_nan_mask] = no_data_value # For all-NaN slices ValueError is raised.
    if DO_DASK:
        import dask.array as da
        from dask.diagnostics import ProgressBar
        # Convert to dask array with specified chunk size
        # 
        dask_stack = da.from_array(arr, chunks=(arr.shape[0], 100, 100))
        
        # Calculate percentile along the first axis (across images)
        # nanpercentile handles NaN values properly
        percentile_values = da.nanquantile(dask_stack, percentile * 1e-2, axis=axis)
        # percentile_values = da.nanpercentile(dask_stack, percentile, axis=axis)
        # index_array = np.abs(dask_stack - percentile_values[np.newaxis, :, :]).argmin(axis=axis)
        index_array = da.nanargmin(np.abs(dask_stack - percentile_values[np.newaxis, :, :]), axis=axis)
        
        # Compute the percentile values (needed for the next step)
        with ProgressBar():
            percentile_values = percentile_values.compute()
            index_array = index_array.compute()
    else:
        # Compute the percentile values while ignoring NaNs
        percentile_values = np.nanpercentile(arr, percentile, axis=axis)
        # index_array = np.abs(arr - percentile_values[np.newaxis, :, :]).argmin(axis=axis)
        index_array = np.nanargmin(np.abs(arr - percentile_values[np.newaxis, :, :]), axis=axis)
    print(type(index_array))
    print(index_array.shape)
    index_array[all_nan_mask] = -1 #no_data_value #-1 # Or some other indicator if invalid .. @Qiang this results in 0 values for pixels that should be nodata in output?
    return index_array, all_nan_mask

def safe_nanarg_stat(arr, stat='max', axis=0):
    """
    Applies nanargmax safely to prevent `ValueError` in all-NaN slices.
    
    Parameters:
    - arr (np.array): Numpy array to compute maximum index ignoring NaNs.
    - axis (int): Axis along which to find the maximum index.
    
    Returns:
    - np.array: Indices of the maximum values along the given axis, avoiding all-NaN slices.
    """
    # Check where rows/columns are all NaNs along a specified axis
    all_nan_mask = np.all(np.isnan(arr), axis=axis)

    if stat == 'max':
        # Replace all-NaN slices with a fill value (e.g., a large negative number)
        fill_value = np.min(arr) if np.min(arr) < 0 else -np.inf
    if stat == 'min':
         fill_value = np.max(arr) if np.max(arr) > 10 else np.inf # <--- @Qiang might have to check this
    # Choose an appropriate fill value based on your data context
    arr_filled = np.where(all_nan_mask[np.newaxis, ...], fill_value, arr)

    if stat == 'max':
        # Return indices, using nanargmax safely
        return np.nanargmax(arr_filled, axis=axis), all_nan_mask
    elif stat == 'min':
        return np.nanargmin(arr_filled, axis=axis), all_nan_mask
    else:
        raise ValueError("Invalid statistic. Choose from 'min', 'max'.")

def compute_stat_from_masked_array(masked_array, 
                                            no_data_value = None, 
                                            stat='max', percentile_value=None):
    """
    Compute the pixel-wise statistic from a numpy masked 3D array, accounting for NaN values and a no data value,
    and return a 2D array.
    
    Parameters:
    - masked_array (np.ma.MaskedArray): A masked 3D array.
    - no_data_value (scalar): The value to mask out from the data before computing statistics.
    - stat (str): The statistic to compute - 'min', 'max', or 'percentile'.
    - percentile_value (float): The percentile value to compute if stat is 'percentile'.
    
    Returns:
    - np.array: A 2D array containing the computed statistic for each pixel.
    """
    
    if not isinstance(masked_array, np.ma.MaskedArray):
        raise ValueError("Input must be a numpy masked array.")
    
    data = np.ma.filled(masked_array, np.nan)  # Convert masked values to NaN
    #print_array_stats(data)
    # Create a mask for the no data value
    #additional_mask = (data == no_data_value)

    if no_data_value is not None:
        print("\tApply the mask for no data values...")
        data = np.ma.masked_array(data, mask=(data == no_data_value))
        data = np.ma.filled(data, np.nan)  # Convert the new mask to NaN
        #print_array_stats(data)
    if stat == 'geomedian':
        print('Not yet implemented: https://github.com/daleroberts/hdmedians/tree/master')
    elif stat == 'min':
        #result = np.nanargmin(data, axis=0)
        result, all_nan_mask = safe_nanarg_stat(data, stat='min', axis=0) # THIS IS UNTESTED AND LIKELY IS INCORRECT FOR 'min'
    elif stat == 'max':
        if False:
            all_nan_mask = np.all(np.isnan(data), axis=0)
            result = np.nanargmax(data, axis=0)
        else:
            result, all_nan_mask = safe_nanarg_stat(data, stat='max', axis=0)
    elif stat == 'percentile':
        if percentile_value is None:
            raise ValueError("For 'percentile', a percentile_value must be provided.")
        result, all_nan_mask = nanpercentile_index(data, percentile_value, axis=0, DO_DASK=False)
        #result, all_nan_mask = nanpercentile_index(data, percentile_value, axis=0, DO_DASK=False)
        ## @Ali Below not working as expected: [1] cant get this to put the result back together like single process above... also, [2] the runtime seemed just as long...
        #result = multiprocess_nanpercentile_index(data, percentile_value, axis=0, num_processes=27)

    else:
        raise ValueError("Invalid statistic. Choose from 'min', 'median', 'max', 'percentile'.")

    # Print statistical summary of the result array
    print(f"\tStatistical summary of index array for stat={stat}")
    print_array_stats(result)
    
    # if stat == 'percentile':
    #     return result
    return np.ma.masked_array(result, mask=all_nan_mask, fill_value=no_data_value) # debug_test_3 and above
    #return np.ma.masked_array(result, mask=all_nan_mask) # debug_test_2
    #return result # debug_test_1

def get_json(s3path, output):
    '''
    Download a json from S3 to the output directory
    '''
    aws_session = boto3.session.Session()
    s3 = aws_session.resource('s3')
    output_file = os.path.join(output_dir, os.path.basename(s3path))
    #TODO split the bucket name from the s3 path
    bucket_name = s3path.split("/")[2]
    s3_key = "/".join(samples3.split("/")[3:])
    s3.Bucket(bucket_name).download_file(s3_key, output_file)
    
    with open(output_file) as f:
        catalog = json.load(f) 
    return catalog
    
def GetBandLists(inJSON, bandnum, comp_type):
    
    BandList = []
    with open(inJSON) as f:
        response = json.load(f)
        
    for i in range(len(response['features'])):
        
        if comp_type=='HLS':
            # Get the HLS product type and the product-specific bands from each feature of the JSON
            product_type = response['features'][i]['id'].split('.')[1]
        elif comp_type=='LC2SR':
            # Get the Landsat product type and the product-specific bands from each feature of the JSON
            product_type = response['features'][i]['id'].split('_')[0]
        else:
            print(f"comp type ({comp_type}) not recognized")
            os._exit(1)

        if product_type=='L30':
            bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
        elif product_type=='S30':
            bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B8A', 6:'B11', 7:'B12',8:'Fmask'})
        ## LC2SR ##
        # TODO: configure cloud mask in CreateNDVIstack_LC2SR() using cloud_qa
        elif product_type=='LT04':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'cloud_qa'})
        elif product_type=='LT05':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'cloud_qa'})
        elif product_type=='LE07':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'cloud_qa'})
        elif product_type=='LC08':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'qa_pixel'})
        elif product_type=='LC09':
            bands = dict({2:'blue', 3:'green', 4:'red', 5:'nir08', 6:'swir16', 7:'swir22', 8:'qa_aerosol'})
        else:
            print("HLS product type not recognized: Must be L30 or S30.")
            os._exit(1)
            
        try:
            #print(f'GetBandLists: {comp_type} {product_type} {bands[bandnum]}')
            getBand = response['features'][i]['assets'][bands[bandnum]]['href']
            # check 's3' is at position [:2]
            if getBand.startswith('s3', 0, 2):
                BandList.append(getBand)
        except Exception as e:
            print(e)
                
    return BandList

def HLS_MASK(ma_fmask, 
             MASK_LIST=['cloud', 'adj_cloud', 'cloud shadow', 'snowice', 'water', 'aerosol_high'], 
             HLS_QA_BIT = {'cirrus': 0, 'cloud': 1, 'adj_cloud': 2, 'cloud shadow':3, 'snowice':4, 'water':5, 'aerosol_l': 6, 'aerosol_h': 7}):
    
    '''This function takes the HLS Fmask layer as a masked array and exports the desired mask image array. 
        The mask_list assigns the QA conditions you would like to mask.
        The default mask_list setting is coded for a vegetation application, so it also removes water and snow/ice.
        See HLS user guide for more details: https://lpdaac.usgs.gov/documents/1326/HLS_User_Guide_V2.pdf
    '''
    
    arr = ma_fmask.data
    msk = np.zeros_like(arr)#.astype(np.bool)
    for m in MASK_LIST:
        if m in HLS_QA_BIT.keys():
            msk += ((arr & 1 << HLS_QA_BIT[m]) ) > 0
        if m == 'aerosol_high':
            msk += ((arr & (1 << HLS_QA_BIT['aerosol_h'])) > 0) * ((arr & (1 << HLS_QA_BIT['aerosol_l'])) > 0)
        if m == 'aerosol_moderate':
            msk += ((arr & (1 << HLS_QA_BIT['aerosol_h'])) > 0) * ((arr | (1 << HLS_QA_BIT['aerosol_l'])) != arr)
        if m == 'aerosol_low':
            msk += ((arr | (1 << HLS_QA_BIT['aerosol_h'])) != arr) * ((arr & (1 << HLS_QA_BIT['aerosol_l'])) > 0)
    return msk > 0

def LC2SR_MASK(ma_cloudqa, 
             MASK_LIST=['cloud', 'adj_cloud', 'cloud shadow', 'snow', 'water']):
    '''This function takes the LC2SR CLOUD_QA layer as a masked array and exports the desired mask image array. 
        The mask_list assigns the QA conditions you would like to mask.
        The default mask_list setting is coded for a vegetation application, so it also removes water and snow.
        See LC2SR user guide for more details: <TODO: get user guide>
    '''
    LC2SR_QA_BIT = {'fill': 0,
                'dilated cloud': 1,
                'cirrus': 2,
                'cloud':3,
                'cloud shadow':4,
                'snow':5,
                'clear': 6,
                'water': 7,
                'cloud confidence l': 8,
                'cloud confidence h': 9,
                'cloud shadow confidence l': 10,
                'cloud shadow confidence h': 11,
                'snow/ice confidence l': 12,
                'snow/ice confidence h': 13,
                'cirrus confidence l': 14,
                'cirrus confidence h': 15
                }
    arr = ma_cloudqa.data
    msk = np.zeros_like(arr)#.astype(np.bool)
    MASK_LIST = [x.lower() for x in MASK_LIST]
    for m in MASK_LIST:
        if m in LC2SR_QA_BIT.keys():
            msk += (arr & 1 << LC2SR_QA_BIT[m]) > 0
        if m.endswith("high"):
            l_bit = m.replace("high", 'l')
            h_bit = m.replace("high", 'h')
            msk += ((arr & (1 << LC2SR_QA_BIT[h_bit])) > 0) * ((arr & (1 << LC2SR_QA_BIT[l_bit])) > 0)
        if m.endswith("moderate"):
            l_bit = m.replace("moderate", 'l')
            h_bit = m.replace("moderate", 'h')
            msk += ((arr & (1 << LC2SR_QA_BIT[h_bit])) > 0) * ((arr | (1 << LC2SR_QA_BIT[l_bit])) != arr)
        if m.endswith("low"):
            l_bit = m.replace("low", 'l')
            h_bit = m.replace("low", 'h')
            msk += ((arr | (1 << LC2SR_QA_BIT[h_bit])) != arr) * ((arr & (1 << LC2SR_QA_BIT[l_bit])) > 0)
    ma_cloudqa.mask += msk > 0 # With +, this will be the union of the various bit masks
    return ma_cloudqa

def MaskArrays(file, in_bbox, height, width, comp_type, epsg="epsg:4326", dst_crs="epsg:4326", incl_trans=False, do_mask=False):
    '''Read a window of data from the raster matching the tile bbox
        Return a masked array for the window (subset) of the input file
        or
        Return the image crs and transform (incl_trans=True).
        Note: could be renamed to Get_MaskArray_Subset()
    '''
    
    with COGReader(file) as cog:
        img = cog.part(in_bbox, bounds_crs=epsg, max_size=None, dst_crs=dst_crs, height=height, width=width)
    if incl_trans:
        return img.crs, img.transform
    
    if comp_type=="HLS":
        if do_mask:
            # Returns the integer Fmask whose bits can be converted to a datamask
            #return (np.squeeze(img.as_masked().astype(int)) )
            return (np.squeeze(img.array.astype(int)) )
        else:
            #return (np.squeeze(img.as_masked().astype(np.float32)) * 0.0001)
            return (np.squeeze(img.array.astype(np.float32)) * 0.0001)
        
    elif comp_type=="LC2SR":
        if do_mask:
            # Returns the integer Fmask whose bits can be converted to a datamask
            #return (np.squeeze(img.as_masked().astype(int)) )
            return (np.squeeze(img.array.astype(int)) )
        else:
            # Surface reflectance collection 2 scaling offset (0.0000275) and bias (- 0.2)
            #return (np.squeeze(img.as_masked().astype(np.float32)) * 0.0000275) - 0.2
            return (np.squeeze(img.array.astype(np.float32)) * 0.0000275) - 0.2
    else:
        print("composite type not recognized")
        os._exit(1)

def create_target_stack(target_spectral_index_name, first_band, second_band, fmask, in_bbox, epsg, dst_crs, height, width, comp_type, rangelims_red = [0.01, 0.1], nodatavalue=-9999):
    '''Calculate stack of target spectral index for each source scene
    Mask out pixels above or below the red band reflectance range limit values'''
    
    second_band_ma =   MaskArrays(second_band, in_bbox, height, width, comp_type, epsg, dst_crs)
    first_band_ma =    MaskArrays(first_band, in_bbox, height, width, comp_type, epsg, dst_crs)
    fmaskarr = MaskArrays(fmask, in_bbox, height, width, comp_type, epsg, dst_crs, do_mask=True)
    
    if comp_type == 'LC2SR':
        fmaskarr =    LC2SR_MASK(fmaskarr)
    else:
        #fmaskarr_by = HLS_MASK(fmaskarr, MASK_LIST=['cloud', 'adj_cloud', 'cloud shadow', 'snowice', 'water', 'aerosol_high']) # mask out snow
        fmaskarr_by = HLS_MASK(fmaskarr, MASK_LIST=['cloud', 'adj_cloud', 'cloud shadow', 'water', 'aerosol_high']) # keep snow

    if target_spectral_index_name == 'evi':
        print(' --- EVI not yet implemented ; using NDVI instead ---')
        target_spectral_index_name = 'ndvi'
    if target_spectral_index_name == 'ndvi':
        print(f'min, max Red value before mask: {first_band_ma.min()}, {first_band_ma.max()} (red rangelims: {rangelims_red})')
        return np.ma.array(np.where(((fmaskarr_by==1) | (first_band_ma < rangelims_red[0]) | (first_band_ma > rangelims_red[1])), nodatavalue, (second_band_ma-first_band_ma)/(second_band_ma+first_band_ma)))
    if target_spectral_index_name == 'ndsi':
        return np.ma.array(np.where((fmaskarr_by==1), nodatavalue, (first_band_ma-second_band_ma)/(first_band_ma+second_band_ma))) # not the same order as ndvi, evi

# def CreateNDVIstack_HLS(REDfile, NIRfile, fmask, in_bbox, epsg, dst_crs, height, width, comp_type, rangelims_red = [0.01, 0.1], nodatavalue=-9999):
#     '''Calculate NDVI for each source scene
#     Mask out pixels above or below the red band reflectance range limit values'''
    
#     NIRarr =   MaskArrays(NIRfile, in_bbox, height, width, comp_type, epsg, dst_crs)
#     REDarr =   MaskArrays(REDfile, in_bbox, height, width, comp_type, epsg, dst_crs)
#     fmaskarr = MaskArrays(fmask, in_bbox, height, width, comp_type, epsg, dst_crs, do_mask=True)
    
#     #
#     # HLS masking
#     #
#     #fmaskarr_by = HLS_MASK(fmaskarr, MASK_LIST=['cloud', 'adj_cloud', 'cloud shadow', 'snowice', 'water', 'aerosol_high']) # mask out snow
#     fmaskarr_by = HLS_MASK(fmaskarr, MASK_LIST=['cloud', 'adj_cloud', 'cloud shadow', 'water', 'aerosol_high']) # keep snow
    
#     #print(f'printing fmaskarr data:\n{fmaskarr.data}')
#     #print(f'printing fmaskarr mask:\n{fmaskarr.mask}')
#     #ndvi = np.ma.array((NIRarr-REDarr)/(NIRarr+REDarr))
#     #print(ndvi.shape)
    
#     print(f'min, max Red value before mask: {REDarr.min()}, {REDarr.max()} (red rangelims: {rangelims_red})')
#     return np.ma.array(np.where(((fmaskarr_by==1) | (REDarr < rangelims_red[0]) | (REDarr > rangelims_red[1])), nodatavalue, (NIRarr-REDarr)/(NIRarr+REDarr)))
    
# def CreateNDVIstack_LC2SR(REDfile, NIRfile, fmask, in_bbox, epsg, dst_crs, height, width, comp_type, rangelims_red = [0.01, 0.1], nodatavalue=-9999):
#     '''Calculate NDVI for each source scene'''
#     NIRarr =   MaskArrays(NIRfile, in_bbox, height, width, comp_type, epsg, dst_crs)
#     REDarr =   MaskArrays(REDfile, in_bbox, height, width, comp_type, epsg, dst_crs)
#     fmaskarr = MaskArrays(fmask, in_bbox, height, width, comp_type, epsg, dst_crs, do_mask=True)
    
#     #
#     # LC2SR masking
#     #
#     fmaskarr = LC2SR_MASK(fmaskarr)

#     # print(f'\tmin, max Red value before mask: {round(REDarr.min(), 4)}, {round(REDarr.max(), 4)} (red rangelims: {rangelims_red})')
#     print(f'\tmin, max Red value before mask: {REDarr.min()}, {REDarr.max()} (red rangelims: {rangelims_red})')
#     #return np.ma.array((NIRarr-REDarr)/(NIRarr+REDarr))
#     return np.ma.array(np.where(((fmaskarr==1) | (REDarr < rangelims_red[0]) | (REDarr > rangelims_red[1])), nodatavalue, (NIRarr-REDarr)/(NIRarr+REDarr)))

def CollapseBands(inArr, NDVItmp, BoolMask, nodatavalue):
    '''
    Inserts the bands as arrays (made earlier)
    Creates a single layer by using the binary mask and a sum function to collapse n-dims to 2-dims
    '''
    inArr = np.ma.masked_equal(inArr, 0)
    inArr[np.logical_not(NDVItmp)]=0 
    compImg = np.ma.masked_array(inArr.sum(0), BoolMask)
    
    return compImg.filled(nodatavalue) # doing this prevents nodata from being returned as 0 values in final composite

def CreateComposite(file_list, NDVItmp, BoolMask, in_bbox, height, width, epsg, dst_crs, comp_type, nodatavalue):
    #print("\t\tMaskedFile")
    MaskedFile = [MaskArrays(file_list[i], in_bbox, height, width, comp_type, epsg, dst_crs) for i in range(len(file_list))]
    #print("\t\tComposite")
    Composite = CollapseBands(MaskedFile, NDVItmp, BoolMask, nodatavalue)
    return Composite

def createJulianDateLC2SR(file, height, width):
    date_string = file.split('/')[-1].split('_')[3]
    fmt = '%Y.%m.%d'
    date = date_string[:4] + '.' + date_string[4:6] + '.' + date_string[6:]
    dt = datetime.datetime.strptime(date, fmt)
    tt = dt.timetuple()
    jd = tt.tm_yday
    date_arr = np.full((height, width), jd,dtype=np.float32)
    return date_arr
    
def JulianCompositeLC2SR(file_list, NDVItmp, BoolMask, height, width, nodatavalue):
    JulianDateImages = [createJulianDateLC2SR(file_list[i], height, width) for i in range(len(file_list))]
    JulianComposite = CollapseBands(JulianDateImages, NDVItmp, BoolMask, nodatavalue)
    return JulianComposite

def createJulianDateHLS(file, height, width):
    j_date = file.split('/')[-1].split('.')[3][4:7]
    date_arr = np.full((height, width),j_date,dtype=np.float32)
    return date_arr
    
def JulianCompositeHLS(file_list, NDVItmp, BoolMask, height, width, nodatavalue):
    JulianDateImages = [createJulianDateHLS(file_list[i], height, width) for i in range(len(file_list))]
    JulianComposite = CollapseBands(JulianDateImages, NDVItmp, BoolMask, nodatavalue)
    return JulianComposite

def JulianComposite(file_list, NDVItmp, BoolMask, height, width, comp_type, nodatavalue):
    if comp_type == 'LC2SR':
        JulianDateImages = [createJulianDateLC2SR(file_list[i], height, width) for i in range(len(file_list))]
    elif comp_type == 'HLS':
        JulianDateImages = [createJulianDateHLS(file_list[i], height, width) for i in range(len(file_list))]
    JulianComposite = CollapseBands(JulianDateImages, NDVItmp, BoolMask, nodatavalue)
    return JulianComposite

def year_band(file, height, width, comp_type):
    if comp_type == "HLS":
        year = file.split('/')[-1].split('.')[3][0:4]
    elif comp_type == "LC2SR":
        year = file.split('/')[-1].split('_')[3][0:4]
        
    year_arr = np.full((height, width),year,dtype=np.float32)
    
    return year_arr

def year_band_composite(file_list, NDVItmp, BoolMask, height, width, comp_type, nodatavalue):
    year_imgs = [year_band(file_list[i], height, width, comp_type) for i in range(len(file_list))]
    year_composite = CollapseBands(year_imgs, NDVItmp, BoolMask, nodatavalue)
    return year_composite

# Snow Index Calculation
# https://www.usgs.gov/landsat-missions/normalized-difference-snow-index

def calcNDSI(green, swir):
    ndsi = ((green - swir) / (green + swir))
    print('\tNDSI Created')
    return ndsi
    
# Vegetation Indices Calculations
# https://www.usgs.gov/landsat-missions/landsat-surface-reflectance-derived-spectral-indices

# NDVI
def calcNDVI(red, nir):
    ndvi = (nir - red)/(nir + red )
    print('\tNDVI Created')
    return ndvi
    
# SAVI
def calcSAVI(red, nir):
    savi = ((nir - red)/(nir + red + 0.5))*(1.5)
    print('\tSAVI Created')
    return savi

# MSAVI
def calcMSAVI(red, nir):
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    print('\tMSAVI Created')
    return msavi

# NDMI
def calcNDMI(nir, swir):
    ndmi = (nir - swir)/(nir + swir)
    print('\tNDMI Created')
    return ndmi

# EVI
def calcEVI(blue, red, nir):
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    print('\tEVI Created')
    return evi

# NBR
def calcNBR(nir, swir2):
    nbr = (nir - swir2)/(nir + swir2)
    print('\tNBR Created')
    return nbr

# NBR2
def calcNBR2(swir, swir2):    
    nbr2 = (swir - swir2)/(swir + swir2)
    print('\tNBR2 Created')
    return nbr2

def tasseled_cap(bands):
    '''
    Compute the tasseled cap indices: brightness, greenness, wetness
    bands - a 6-layer 3-D (images) or 2-D array (samples) or an OrderedDict with appropriate band names
    tc_coef - a list of 3 tuples, each with 6 coefficients
    '''
    # Tasseled Cap (At Satellite)
    #tc_coef = [
    #(0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872), #brightness
    #(-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608), #greenness
    #(0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559) #wetness
    #]
    
    # Tasseled Cap (SREF: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147121#sec028)
    tc_coef = [
    (0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303), #brightness
    (-0.1603, 0.2819, -0.4934, 0.7940, -0.0002, -0.1446), #greenness
    (0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109) #wetness
    ]
    
    tc = np.zeros((len(np.shape(bands)), bands.shape[1], bands.shape[2]), dtype = np.float32())
    
    #print(np.max(in_arr_sref))
    for i, t in enumerate(tc_coef):
        for b in range(5): # should be 6
            tc[i] += (bands[b] * t[b]).astype(np.float32())
           
    print('\tTassCap Created')
    return tc[0], tc[1], tc[2] 

    # TC Code adapted from: https://github.com/bendv/waffls/blob/master/waffls/indices.py
    # TC coeffs from: https://doi.org/10.1080/2150704X.2014.915434 (OLD at satellite coeffs)
    # New coeffs are in sup table 2 here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147121#sec028
    # which in turn are from/built from: Crist 1985. These are sensor non-specific so should be applicable
    # irrespective of sensor and collection, provided it is SREF

def VegMask(NDVI, MIN_NDVI = 0.1, NODATAVALUE=-9999):
    print(f"Creating binary vegetation mask where NDVI > {MIN_NDVI} indicates valid data, set to 1, all else 0...")
    print(f'\tmin NDVI value before mask: {np.nanmin(np.where(NDVI == NODATAVALUE, np.nan, NDVI))}')
    mask = np.zeros_like(NDVI)
    mask = np.where(NDVI > MIN_NDVI, 1, mask)
    return mask

def get_pixel_coords(arr, transform):
    rows = np.arange(0,np.shape(arr)[0],1)
    Yarr = (((transform[2]+(0.5*transform[0])) + (rows*transform[0])).reshape(np.shape(arr)[0],1))[::-1]
    Ygeo = np.tile(Yarr, np.shape(arr)[1]).astype(np.float32())
    cols = np.arange(0,np.shape(arr)[1],1)
    Xarr = ((transform[5]+(0.5*transform[4])) + (cols*transform[4]))[::-1]
    Xgeo = np.tile(Xarr, (np.shape(arr)[0],1)).astype(np.float32())
    
    return Xgeo, Ygeo

def renew_session(comp_type):
    if comp_type == 'HLS':
        aws_session = get_aws_session_DAAC(get_creds_DAAC())
    elif comp_type == 'LC2SR':
        aws_session = get_aws_session(get_creds())
    return aws_session   

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_tile_fn", type=str, default="/projects/shared-buckets/montesano/databank/boreal_tiles_v004.gpkg", help="The filename of the stack's set of vector tiles")
    parser.add_argument("-n", "--in_tile_num", type=int, help="The id of a tile that will define the bounds of the raster stacking")
    parser.add_argument("-o", "--output_dir", type=str, help="The path for the JSON files to be written")
    parser.add_argument("-b", "--tile_buffer_m", type=float, default=0, help="The buffer size (m) applied to the extent of the specified stack tile")
    parser.add_argument("-r", "--res", type=int, default=30, help="The output resolution of the stack")
    parser.add_argument("--shape", type=int, default=None, help="The output height and width of the grid's shape. If None, get from input tile.")    
    parser.add_argument("-lyr", "--in_tile_layer", type=str, default=None, help="The layer name of the stack tiles dataset")
    parser.add_argument("-in_tile_id_col", type=str, default="tile_num", help="The column of the tile layer name of the stack tiles dataset that holds the tile num")
    parser.add_argument("-a", "--sat_api", type=str, default="https://cmr.earthdata.nasa.gov/stac/LPCLOUD", help="URL of API to query HLS archive")
    parser.add_argument("-j", "--json_file", type=str, default=None, help="The S3 path to the query response json")
    parser.add_argument("-l", "--local", type=bool, default=False, help="Dictate whether it is a run using local paths")
    parser.add_argument("-sy", "--start_year", type=str, default="2020", help="specify the start year date (e.g., 2020)")
    parser.add_argument("-ey", "--end_year", type=str, default="2021", help="specify the end year date (e.g., 2021)")
    parser.add_argument("-smd", "--start_month_day", type=str, default="06-01", help="specify the start month and day (e.g., 06-01)")
    parser.add_argument("-emd", "--end_month_day", type=str, default="09-15", help="specify the end month and day (e.g., 09-15)")
    parser.add_argument("-mc", "--max_cloud", type=int, default=40, help="specify the max amount of cloud")
    parser.add_argument("-t", "--composite_type", choices=['HLS','LC2SR'], nargs="?", type=str, default='HLS', const='HLS', help="Specify the composite type")
    parser.add_argument("--rangelims_red", type=float, nargs=2, action='store', default=[-1e9, 1e9], help="The range limits for red band; outside of which will be masked out; eg [0.01, 0.1]")
    parser.add_argument("-hls", "--hls_product", choices=['S30','L30','H30'], nargs="?", type=str, default='L30', help="Specify the HLS product; M30 is our name for a combined HLS composite")
    parser.add_argument("-hlsv", "--hls_product_version", type=str, default='2.0', help="Specify the HLS product version")
    parser.add_argument("--target_spectral_index", type=str, choices=['ndvi','ndsi'], nargs="?", default="ndvi", help="The target spectral index used with stat to composite a stack of input.")
    parser.add_argument("-ndvi", "--thresh_min_ndvi", type=float, default=0.1, help="NDVI threshold above which vegetation is valid.")
    parser.add_argument("-min_n", "--min_n_filt_results", type=int, default=0, help="Min number of filtered search results desired before hitting max cloud limit.")
    parser.add_argument("--stat", type=str, choices=['min','max','percentile'], nargs="?", default="max", help="Specify the stat for reducing the NDVI stack")
    parser.add_argument("--stat_pct", type=float, default=98.0, help="Specify the specific percentile stat for reducing the NDVI stack")
    parser.add_argument('--do_indices', dest='do_indices', action='store_true', help='Compute all the spectral indices.')
    parser.add_argument('--search_only', dest='search_only', action='store_true', help='Only perform search and return response json. No composites made.')
    parser.set_defaults(search_only=False)
    parser.add_argument("-ndv", "--nodatavalue", type=int, default=-9999, help="No data value.")
    args = parser.parse_args()    
    
    '''
    Build multi-spectral (ms) composites with scenes from queries of:
    (a) An endpoint of the USGS Landsat-5/7/8 archive
    (b) An endpoint of the v 2.0 of the HLS S30, L30 archive
    
    The ms composite will have the following bands:
    
    'Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIR2': surface reflectance values for ms composite obs.
    'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2' : indices calc'd from the surface reflectance values of the ms composite obs.
    'TCB', 'TCG', 'TCW' : tasseled cap values from the ms composite obs.
    'ValidMask' : a mask identifing valid vegetation obs.
    'Xgeo' : x coordinate
    'Ygeo' : y coordinate
    'JulianDate' : day-of-year of ms composite obs.
    'yearDate' : year of ms composite obs.
    'count': the pixelwise count of valid observations considered during compositing.
    
    Note: 
    HLS info:
    https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/harmonized-landsat-sentinel-2-hls-overview/  
    
    LDPAAC Forum:
    https://forum.earthdata.nasa.gov/viewforum.php?f=7
    
    GitHub CMR issue:
    https://github.com/nasa/cmr-stac/issues
    '''
    if args.do_indices:
        bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIR2', 
                     'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', #NDSI,
                     'ValidMask', 'Xgeo', 'Ygeo', 'JulianDate', 'yearDate','count']
    else:
        bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIR2', 
                     'ValidMask', 'JulianDate', 'yearDate','count', 'Fmask']
    
    geojson_path_albers = args.in_tile_fn
    print('\nTiles path:\t\t', geojson_path_albers)
    tile_n = args.in_tile_num
    print("Tile number:\t\t", tile_n)
    res = args.res
    print("Output res (m):\t\t", res)
    
    tile_buffer_m = args.tile_buffer_m
    
    # Get tile by number form GPKG. Store box and out crs
    tile_id = get_index_tile(vector_path=geojson_path_albers, id_col=args.in_tile_id_col, tile_id=tile_n, buffer=tile_buffer_m, layer = args.in_tile_layer)
    #in_bbox = tile_id['bbox_4326']
    in_bbox = tile_id['geom_orig_buffered'].bounds.iloc[0].to_list()
    out_crs = tile_id['tile_crs']
    
    print("in_bbox:\t\t", in_bbox)
    print('bbox 4326:\t\t', tile_id['bbox_4326'])

    # This is added to allow the output size to be forced to a certain size - this avoids have some tiles returned as 2999 x 3000 due to rounding issues.
    # Most tiles dont have this problem and thus dont need this forced shape, but some consistently do. 
    if args.shape is None:
        print(f'Getting output dims from buffered (buffer={tile_buffer_m}) original tile geometry...')
        height, width = get_shape(in_bbox, res)
    else:
        print('Getting output dims from input shape arg...')
        height = args.shape
        width = args.shape
    
    print(f'Output dims:\t\t{height} x {width}')
    
    print(f'Composite type:\t\t{args.composite_type}')
    
    if args.json_file == None:
        if args.output_dir == None:
            print("MUST SPECIFY -o FOR JSON PATH")
            os._exit(1)
        elif args.composite_type == 'HLS':
            ms_product = args.hls_product
            ms_version = args.hls_product_version
        elif args.composite_type == 'LC2SR':
            ms_product = 'landsat-c2l2-sr'
            ms_version = None
        else:
            print("specify the composite type (HLS, LC2SR)")
            os._exit(1)
            
        # Get the multispectral data for compositing as a json file of CMR query responses
        master_json = get_ms_data(args.in_tile_fn, args.in_tile_layer, args.in_tile_id_col, args.in_tile_num, args.output_dir, args.sat_api, 
                                args.start_year, args.end_year, args.start_month_day, args.end_month_day, args.max_cloud, 
                                args.composite_type, args.local, ms_product, ms_version, args.min_n_filt_results, bands_dict=MS_BANDS_DICT)
    else:
        master_json = args.json_file
    
    if args.search_only:
        print(f"Search only mode. Master JSON written: {master_json}")
        os._exit(1)
    
    blue_bands = GetBandLists(master_json, 2, args.composite_type)
    print(f"\nTotal # of scenes for composite:\t\t{len(blue_bands)}")
    if len(blue_bands) == 0:
            print("\nNo scenes to build a composite. Exiting.\n")
            os._exit(1)  
    green_bands = GetBandLists(master_json, 3, args.composite_type)
    red_bands =   GetBandLists(master_json, 4, args.composite_type)
    nir_bands =   GetBandLists(master_json, 5, args.composite_type)
    swir_bands =  GetBandLists(master_json, 6, args.composite_type)
    swir2_bands = GetBandLists(master_json, 7, args.composite_type)
    fmask_bands = GetBandLists(master_json, 8, args.composite_type)

    print(f'Creating {args.target_spectral_index} stack with {args.composite_type} ...')

    # insert AWS credentials here if needed
    if args.composite_type == 'HLS':
        aws_session = get_aws_session_DAAC(get_creds_DAAC())
    elif args.composite_type == 'LC2SR':
        aws_session = get_aws_session(get_creds())
    else:
        print("specify the composite type (HLS, LC2SR)")
        os._exit(1)
    
    #print(aws_session)
    
    ###############
    # After the choice of 'args.target_spectral_index', the order in which bands are fed into the 'Create Stack' function is important:
    # this function takes bands in as args in the order in which the bands appear on the EO spectrum, not the order in which they appear in the spectral index.
    # so, while ndvi = (nir-red) / (nir+red) ; the 'reate Stack' function wants first the red_bands, then the nir_bands
    
    # Start reading data on aws:
    with rio.Env(aws_session):
        in_crs, crs_transform = MaskArrays(red_bands[0], in_bbox, height, width, args.composite_type, out_crs, out_crs, incl_trans=True)
        # if args.composite_type=='HLS':
        #     target_spec_idx_stack = [CreateNDVIstack_HLS(red_bands[i],nir_bands[i],fmask_bands[i], 
        #                                      in_bbox, out_crs, out_crs, height, width, 
        #                                      args.composite_type, rangelims_red = args.rangelims_red) for i in range(len(red_bands))]
        # elif args.composite_type=='LC2SR':
        #     target_spec_idx_stack = [CreateNDVIstack_LC2SR(red_bands[i],nir_bands[i],fmask_bands[i], 
        #                                        in_bbox, out_crs, out_crs, height, width, 
        #                                        args.composite_type, rangelims_red = args.rangelims_red) for i in range(len(red_bands))]

        # These are ordered from smallest to largest wavelength 
        if args.target_spectral_index == 'ndsi': first_bands, second_bands = (green_bands, swir_bands)
        if args.target_spectral_index == 'ndvi': first_bands, second_bands = (red_bands, nir_bands)
            
        target_spec_idx_stack = [create_target_stack(args.target_spectral_index, first_bands[i], second_bands[i], fmask_bands[i], 
                                                     in_bbox, out_crs, out_crs, height, width, args.composite_type, rangelims_red = args.rangelims_red, nodatavalue=args.nodatavalue) for i in range(len(red_bands))]
        
        print(f'\nFinished created masked {args.target_spectral_index} stack.\n')
       
    ###############
    print(f"Make {args.target_spectral_index} masked array")
    target_spec_idx_stack_ma = np.ma.array(target_spec_idx_stack)
    print("shape:\t\t", target_spec_idx_stack_ma.shape)
    
    print(f"\nCalculating index positions from {args.target_spectral_index} stack using stat: {args.stat} (w/ nodatavalue = {args.nodatavalue})...")
    target_stat = compute_stat_from_masked_array(target_spec_idx_stack_ma, args.nodatavalue, stat=args.stat, percentile_value=args.stat_pct)
    BoolMask = np.ma.getmask(target_stat)
    # create a tmp array (binary mask) of the same input shape
    SPEC_IDX_STK_tmp = np.ma.zeros(target_spec_idx_stack_ma.shape, dtype=bool)
    
    # Get the pixelwise count of the valid data
    CountComp = np.sum((target_spec_idx_stack_ma != -9999), axis=0)
    print(f"Count array min ({CountComp.min()}), max ({CountComp.max()}), and shape ({CountComp.shape})")
    
    # for each dimension assign the index position (flattens the array to a LUT)
    print(f"Create LUT of {args.target_spectral_index} positions using stat={args.stat}")
    for i in range(np.shape(target_spec_idx_stack_ma)[0]):
        SPEC_IDX_STK_tmp[i,:,:]=target_stat==i
        
    ##############
    # kw_args = {'SPEC_IDX_STK_tmp': SPEC_IDX_STK_tmp, 'BoolMask': BoolMask, 'in_bbox': in_bbox, 'height': height, 'width': width, 'out_crs': out_crs, 'composite_type': args.composite_type, 'nodatavlue': args.nodatavalue}
    # params_list = [
    #     {'band_name': 'Blue', 'bands_list': blue_bands},
    #     {'band_name': 'Green', 'bands_list': green_bands},
    #     {'band_name': 'Red', 'bands_list': red_bands},
    #     {'band_name': 'NIR', 'bands_list': nir_bands},
    #     {'band_name': 'SWIR', 'bands_list': swir_bands},
    #     {'band_name': 'SWIR2', 'bands_list': swir2_bands},
    #     {'band_name': 'Julian Day', 'bands_list': swir2_bands},
    #     {'band_name': 'Year', 'bands_list': swir2_bands},
    #     {'band_name': 'Fmask', 'bands_list': fmask_bands},
    # ]
    # def wrapper_createcomposite(params):
    #     aws_session = renew_session(params['composite_type'])

    #     with rio.Env(aws_session):
    #         print(f'Creating {params['band_name']} composite...')
    #         _comp = CreateComposite(params['bands_list'], params['SPEC_IDX_STK_tmp'], params['BoolMask'], params['in_bbox'], params['height'], params['width'], params['out_crs'], params['out_crs'], params['composite_type'], params['nodatavalue'])
    #         #print_array_stats(_comp)
    #         return _comp

    # from multiprocessing import Pool
    # from functools import partial
    
    # with Pool(processes=6) as pool:
    #     BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, SWIR2Comp, JULIANcomp, YEARcomp, Fmaskcomp = pool.map(partial(wrapper_createcomposite, **kw_args), params_list)
            
    # create band-by-band composites: TODO multiprocess these
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Blue composite...')
        BlueComp = CreateComposite(blue_bands, SPEC_IDX_STK_tmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type, args.nodatavalue)
        #print_array_stats(BlueComp)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Green composite...')
        GreenComp = CreateComposite(green_bands, SPEC_IDX_STK_tmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type, args.nodatavalue)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Red composite...')
        RedComp = CreateComposite(red_bands, SPEC_IDX_STK_tmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type, args.nodatavalue)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating NIR composite...')
        NIRComp = CreateComposite(nir_bands, SPEC_IDX_STK_tmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type, args.nodatavalue)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating SWIR composite...')
        SWIRComp = CreateComposite(swir_bands, SPEC_IDX_STK_tmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type, args.nodatavalue)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating SWIR2 composite...')
        SWIR2Comp = CreateComposite(swir2_bands, SPEC_IDX_STK_tmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type, args.nodatavalue)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session): 
        print('Creating Julian Date composite...')
        JULIANcomp = JulianComposite(swir2_bands, SPEC_IDX_STK_tmp, BoolMask, height, width, args.composite_type, args.nodatavalue)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Year Date composite...')
        YEARcomp = year_band_composite(swir2_bands, SPEC_IDX_STK_tmp, BoolMask, height, width, args.composite_type, args.nodatavalue)
    aws_session = renew_session(args.composite_type)
    with rio.Env(aws_session):
        print('Creating Fmask composite...')
        Fmaskcomp = CreateComposite(fmask_bands, SPEC_IDX_STK_tmp, BoolMask, in_bbox, height, width, out_crs, out_crs, args.composite_type, args.nodatavalue)

    DO_VALIDMASK_FROM_NDVI = False
    if args.do_indices or args.target_spectral_index == 'ndvi': # NDVIComp will be required in this case
        DO_VALIDMASK_FROM_NDVI = True
        
        # Originally calc'd NDVI like this - now using RedComp and NIRComp which come out of CollapseBands() at the end of CreateComposite()
        # aws_session = renew_session(args.composite_type)
        # with rio.Env(aws_session):
        #     print('Creating NDVI composite for a valid vegation mask...')
        #     NDVIComp = CollapseBands(NDVIstack_ma, NDVItmp, BoolMask, args.nodatavalue)
        NDVIComp =  calcNDVI(RedComp, NIRComp)

    if DO_VALIDMASK_FROM_NDVI:
        print(f"\nGenerating a valid mask using min NDVI threshold ({args.thresh_min_ndvi}) on NDVI composite...")
        ValidMask = VegMask(NDVIComp, MIN_NDVI=args.thresh_min_ndvi)
    else:
        print("\nNo additional masking of valid values based on ndvi threshold...(valid mask all 1's)")
        ValidMask = np.ones_like(BlueComp)
    
    if args.do_indices:
        print("\nGenerating spectral indices...")
        SAVI =  calcSAVI(RedComp, NIRComp)
        MSAVI = calcMSAVI(RedComp, NIRComp)
        NDMI =  calcNDMI(NIRComp, SWIRComp)
        EVI =   calcEVI(BlueComp, RedComp, NIRComp)
        NBR =   calcNBR(NIRComp, SWIR2Comp)
        NBR2 =  calcNBR2(SWIRComp, SWIR2Comp)
        TCB, TCG, TCW = tasseled_cap(np.transpose([BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, SWIR2Comp], [0, 1, 2]))
    
    # Stack bands together
    print("\nCreating raster stack...\n")
    # These must correspond with the bandnames
    if args.do_indices:
        print("Calculating X and Y pixel center coords...")
        Xgeo, Ygeo = get_pixel_coords(ValidMask, crs_transform)
        stack = np.transpose([BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, SWIR2Comp, 
                              NDVIComp, SAVI, MSAVI, NDMI, EVI, NBR, NBR2, TCB, TCG, TCW, 
                              ValidMask, Xgeo, Ygeo, JULIANcomp, YEARcomp, CountComp], [0, 1, 2]) 
    else:
        stack = np.transpose([BlueComp, GreenComp, RedComp, NIRComp, SWIRComp, SWIR2Comp, 
                              ValidMask, JULIANcomp, YEARcomp, CountComp, Fmaskcomp], [0, 1, 2]) 
     
    print(f"Assigning band names:\n\t{bandnames}\n")
    print("specifying output directory and filename")

    outdir = args.output_dir
    start_season = args.start_month_day[0:2] + args.start_month_day[2:]
    end_season = args.end_month_day[0:2] + args.end_month_day[2:]
    start_year = args.start_year
    end_year = args.end_year
    comp_type = args.composite_type
    
    if args.stat != 'percentile': 
        STAT = f'{args.stat}{args.target_spectral_index}'
    else:
        STAT = f'{args.stat}{args.stat_pct}{args.target_spectral_index}'
    out_stack_fn = os.path.join(outdir, '_'.join([comp_type, str(tile_n), start_season, end_season, start_year, end_year, STAT]) + '.tif')
    
    print('\nApply a common mask across all layers of stack...')
    print(f"Stack shape pre-mask:\t\t{stack.shape}")
    stack[:, np.any(stack == args.nodatavalue, axis=0)] = args.nodatavalue 
    print(f"Stack shape post-mask:\t\t{stack.shape}")
    print_array_stats(stack)
    
    # write COG to disk
    write_cog(stack, 
              out_stack_fn, 
              in_crs, 
              crs_transform, 
              bandnames, 
              out_crs=out_crs, 
              resolution=(res, res), 
              align=True, ### Debug
              clip_geom=tile_id["geom_orig"],
              input_nodata_value = args.nodatavalue
             )
    print(f"Wrote out stack:\t\t{out_stack_fn}\n")
    return(out_stack_fn)
    
if __name__ == "__main__":
    main()