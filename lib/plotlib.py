import rasterio
from rasterio.plot import show_hist, show
import numpy as np
import matplotlib.pyplot as plt

def add_colorbar(mappable, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    plt.sca(last_axes)
    return cbar

def rescale_pct_clip(array, pct=[1,80]):
    '''Re-scales data values of an array from 0-1 with percentiles'''
    array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
    clip = (array - array_min) / (array_max - array_min)
    clip[clip>1]=1
    clip[clip<0]=0
    return clip
    
def rescale_abs_clip(array, MIN_MAX_LIST=[(0, 6000),(0, 2000),(0, 2000)]):
    '''Re-scales data values of an array to absolute min and max values'''
    # Apply limits by clipping the data
    b1 = np.clip(array[0,:,:], *MIN_MAX_LIST[0])
    b2 = np.clip(array[1,:,:], *MIN_MAX_LIST[1])
    b3 = np.clip(array[2,:,:], *MIN_MAX_LIST[2])
    clip_arr = np.stack((
        (b1 - MIN_MAX_LIST[0][0]) / (MIN_MAX_LIST[0][1] - MIN_MAX_LIST[0][0]),
        (b2 - MIN_MAX_LIST[1][0]) / (MIN_MAX_LIST[1][1] - MIN_MAX_LIST[1][0]),
        (b3 - MIN_MAX_LIST[2][0]) / (MIN_MAX_LIST[2][1] - MIN_MAX_LIST[2][0])
    ))
    return clip_arr
    
def rescale_multiband_for_plot(fn, rescaled_multiband_fn, bandlist = [4,3,2], pct=[5,95], nodata=-9999.0):
    
    # add a reduced res: https://gis.stackexchange.com/questions/434441/specifying-target-resolution-when-resampling-with-rasterio
    
    with rasterio.open(fn, "r") as src1:
        #print(src1.profile)

        # This works only in r+ mode - which isnt possible when reading from private s3 bucket i think
        #src1.nodata = nodata
        
        arr_list = []
        for band in bandlist:
            arr = src1.read(band)
            arr_list.append(arr)
            
        with rasterio.open(rescaled_multiband_fn, 'w+',
                driver='GTiff',
                dtype= rasterio.float32,
                count=3,
                crs = src1.crs,
                width=src1.width,
                height=src1.height,
                transform=src1.transform,
                nodata=src1.nodata

            ) as dst:

            for i, band in enumerate(bandlist): 
                V = rescale_pct_clip(src1.read(band), pct=pct)
                dst.write(V,i+1)