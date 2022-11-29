

''' Author: Nathan Thomas, Paul Montesano
    Date: 02/003/2020
    Version: 1.0
    THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''

import h5py
#from osgeo import gdal
import numpy as np
import pandas as pd
import subprocess
import os
import math

import argparse

import datetime, time
from datetime import datetime

def rec_merge1(d1, d2):
    '''return new merged dict of dicts'''
    for k, v in d1.items(): # in Python 2, use .iteritems()!
        if k in d2:
            d2[k] = rec_merge1(v, d2[k])
    d3 = d1.copy()
    d3.update(d2)
    return d3

def extract_atl08(args):
    TEST = args.TEST
    seg_length = args.seg_length
    #get_gedi_rh = args.get_gedi_rh
    rh_type = args.rh_type
    filter_qual = args.filter_qual
    
    # File path to ICESat-2h5 file
    H5 = args.input
    
    # Get the filepath where the H5 is stored and filename
    inDir = '/'.join(H5.split('/')[:-1])
    granule_fname = H5.split('/')[-1]
    Name = granule_fname.split('.')[0]

    # Get version
    atl08_version = int(granule_fname.split('_')[-2])
    
    if atl08_version < 5:
        print("\nNeed ATL08 v5 or above to implement the updated v3 filtering.")
        print("The v3 quality filtering from FilterUtils.py wants to use 'seg_cover' which is not available before ATL08 v5")
        print("Build in logic to this script to use v2 quality filtering from FilterUtils.py, which will work with ATL08 v3")
        print("Turning filtering off now.")
        print('Quality Filtering: \t[OFF] (you should upgrade ATL08 to v5)')
        filter_qual = False
    
    if args.output == None:
        outbase = os.path.join(inDir, Name)
    else:
        outbase = os.path.join(args.output, Name)
        
    print("\nATL08 granule name: \t{}".format(Name))
    print("Input dir: \t\t{}".format(inDir))

    if args.overwrite:
        # Overwite is True (on)
        pass
    else:
        if os.path.isfile(os.path.join(outbase + '.csv')):
            # Overwite is False (off) and file exists
            print("FILE EXISTS AND WE'RE NOT OVERWRITING")
            os._exit(1)
        else:
            # Overwite is False (off) but file DOES NOT exist
            pass

    land_seg_path = '/land_segments/'
    if seg_length == 30:
        do_30m = True
        land_seg_path = land_seg_path + str(seg_length) + 'm_segment/'
    else:
        do_30m = False
        
    fn_tail = '_' + str(seg_length) + 'm.csv'
    
    print("\nSegment length: {}m".format(seg_length)) 
    
    # open file
    f = h5py.File(H5,'r')

    # Set up acq date
    dt, yr, m, d = ([] for i in range(4))

    # Set up orbit info fields
    gt, orb_num, rgt, orb_orient = ([] for i in range(4))

    # Set the names of the 6 lasers
    lines = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']

    # set up blank lists
    latitude, longitude, segid_beg, segid_end = ([] for i in range(4))

    # Canopy fields
    can_h_met_0, can_h_met_1, can_h_met_2, can_h_met_3, can_h_met_4, can_h_met_5, can_h_met_6, can_h_met_7, can_h_met_8 = ([] for i in range(9))
    can_h_met = []   # Relative	(RH--)	canopy height	metrics calculated	at	the	following	percentiles: 25,50,	60,	70,	75,	80,	85,	90,	95
    h_max_can = []
    h_can = []      # 98% height of all the individual canopy relative heights for the segment above the estimated terrain surface. Relative canopy heights have been computed by differencing the canopy photon height from the estimated terrain surface.
    h_can_unc = []
    h_can_quad = []

    n_ca_ph = []
    n_toc_ph = []
    can_open = []    # stdv of all photons classified as canopy within segment
    can_rh_conf = [] # Canopy relative height confidence flag based on percentage of ground and canopy photons within a segment: 0 (<5% canopy), 1 (>5% canopy, <5% ground), 2 (>5% canopy, >5% ground)
    tcc_flg = [] # Flag indicating that more than 50% of the Landsat Continuous Cover product have values > 100 for the L-Km segment.  Canopy is assumed present along the L-km segment if landsat_flag is 1.
    tcc_prc = [] # Average percentage value of the valid (value <= 100) Landsat Tree Cover Continuous Fields product for each 100 m segment
    seg_cover = [] # Average percentage value of the valid (value <= 100) Copernicus fractional cover product for each 100 m segment

    # Uncertainty fields
    n_seg_ph = []   # Number of photons within each land segment.
    cloud_flg = []     # Valid range is 0 - 10. Cloud confidence flag from ATL09 that indicates the number of cloud or aerosol layers identified in each 25Hz atmospheric profile. If the flag is greater than 0, aerosols or clouds could be present.
    msw_flg = []    # Multiple Scattering warning flag. The multiple scattering warning flag (ATL09 parameter msw_flag) has values from -1 to 5 where zero means no multiple scattering and 5 the greatest. If no layers were detected, then msw_flag = 0. If blowing snow is detected and its estimated optical depth is greater than or equal to 0.5, then msw_flag = 5. If the blowing snow optical depth is less than 0.5, then msw_flag = 4. If no blowing snow is detected but there are cloud or aerosol layers detected, the msw_flag assumes values of 1 to 3 based on the height of the bottom of the lowest layer: < 1 km, msw_flag = 3; 1-3 km, msw_flag = 2; > 3km, msw_flag = 1. A value of -1 indicates that the signal to noise of the data was too low to reliably ascertain the presence of cloud or blowing snow. We expect values of -1 to occur only during daylight.
    night_flg = []
    seg_landcov = [] # IGBP Land Cover Surface type classification as reference from MODIS Land Cover(ANC18) at the 0.5 arcsecond res. flag_values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 flag_meanings : Water Evergreen_Needleleaf_Forest Evergreen_Broadleaf_Forest Deciduous_Needleleaf_Forest Deciduous_Broadleaf_Forest Mixed_Forest Closed_Shrublands Open_Shrubla

    seg_snow = []  # 0=ice free water; 1=snow free land;  2=snow; 3=ice. Daily snow/ice cover from ATL09 at the 25 Hz rate(275m) indicating likely presence of snow and ice within each segment.
    seg_water = []  # no_water=0, water=1. Water mask(i.e. flag) indicating inland water as referenced from the Global Raster Water Mask(ANC33) at 250 m spatial resolution.
    sig_vert = []    # Total vertical geolocation error due to ranging and local surface slope.  The parameter is computed for ATL08 as described in equation 1.2.
    sig_acr = []       # Total cross-track uncertainty due to PPD and POD knowledge.  Read from ATL03 product gtx/geolocation/sigma_across. Sigma_atlas_y is reported on ATL08 as the uncertainty of the center-most reference photon of the 100m ATL08 segment.
    sig_along = []        # Total along-track uncertainty due to PPD and POD knowledge.  Read from ATL03 product gtx/geolocation/sigma_along. Sigma_atlas_x is reported on ATL08 as the uncertainty of the center-most reference photon of the 100m ATL08 segment.
    sig_h = []            # Estimated uncertainty for the reference photon bounce point ellipsoid height: 1- sigma (m) provided at the geolocation segment rate on ATL03.  Sigma_h is reported on ATL08 as the uncertainty of the center-most reference photon of the 100m ATL08 segment.
    sig_topo = []         # Total uncertainty that include sigma_h plus geolocation uncertainty due to local slope (equation 1.3).  The local slope is multiplied by the geolocation uncertainty factor. This will be used to determine the total vertical geolocation error due to ranging and local slope.

    # Terrain fields
    n_te_ph = []
    h_te_best = []  # The best fit terrain elevation at the the mid-point location of each 100m segment. The mid-segment terrain elevation is determined by selecting the best of three fits- linear, 3rd order and 4th order polynomials - to the terrain photons and interpolating the elevation at the mid-point location of the 100 m segment. For the linear fit, a slope correction and weighting is applied to each ground photon based on the distance to the slope height at the center of the segment.
    h_te_unc = []    # Uncertainty of the mean terrain height for the segment. This uncertainty incorporates all systematic uncertainties(e.g. timing orbits, geolocation,etc.) as well as uncertainty from errors of identified photons.  This parameter is described in section 1, equation 1.4
    ter_slp = []        # The along-track slope of terrain, within each segment;computed by a linear fit of terrain classified photons. Slope is in units of delta height over delta along track distance.
    snr = []        # The signal to noise ratio of geolocated photons as determined by the ratio of the superset of ATL03 signal and DRAGANN found signal photons used for processing the ATL08 segments to the background photons (i.e. noise) within the same ATL08 segments.
    sol_az = []     # The direction, eastwards from north, of the sun vector as seen by an observer at the laser ground spot.
    sol_el = []     # Solar Angle above or below the plane tangent to the ellipsoid surface at the laser spot. Positive values mean the sun is above the horizon, while  negative values mean it is below the horizon. The effect of atmospheric refraction is not included. This is a low precision value, with approximately TBD degree accuracy.

    asr = []		# Apparent surface reflectance
    h_dif_ref = []	# height difference from reference DEM
    ter_flg = []
    ph_rem_flg = []
    dem_rem_flg = []
    seg_wmask = []
    lyr_flg = []

    if False:
        # Granule level info
        granule_dt = datetime.strptime(Name.split('_')[1], '%Y%m%d%H%M%S')

        YEAR = granule_dt.year
        MONTH = granule_dt.month
        DOY = granule_dt.timetuple().tm_yday

        Arrtemp = f['/orbit_info/orbit_number/'][...,]

        temp = np.empty_like(Arrtemp, dtype='a255')
        temp[...] = YEAR
        yr.append(temp)

        temp = np.empty_like(Arrtemp, dtype='a255')
        temp[...] = YEAR
        yr.append(temp)

        temp = np.empty_like(Arrtemp, dtype='a255')
        temp[...] = MONTH
        m.append(temp)

        temp = np.empty_like(Arrtemp, dtype='a255')
        temp[...] = DOY
        d.append(temp)

    dt.append(f['/ancillary_data/granule_end_utc/'][...,].tolist())
    orb_orient.append(f['/orbit_info/sc_orient/'][...,].tolist())
    orb_num.append(f['/orbit_info/orbit_number/'][...,].tolist())
    rgt.append(f['/orbit_info/rgt/'][...,].tolist())

    #yr          =np.array([yr[l][k] for l in range(1) for k in range(len(yr[l]))] )
    #m           =np.array([m[l][k] for l in range(1) for k in range(len(m[l]))] )
    #d           =np.array([d[l][k] for l in range(1) for k in range(len(d[l]))] )
    dt          =np.array([dt[l][k] for l in range(1) for k in range(len(dt[l]))] )    
    orb_orient  =np.array([orb_orient[l][k] for l in range(1) for k in range(len(orb_orient[l]))] )
    orb_num     =np.array([orb_num[l][k] for l in range(1) for k in range(len(orb_num[l]))] )
    rgt         =np.array([rgt[l][k] for l in range(1) for k in range(len(rgt[l]))] )

    # Beam level info
    # For each laser read the data and append to its list
    for line in lines:

        # It might be the case that a specific line/laser has no members in the h5 file.
        # If so, catch the error and skip - MW 3/31
        try:
            latitude.append(f['/' + line    + land_seg_path + 'latitude/'][...,].tolist())
        except KeyError:
            continue # No info for laser/line, skip it and move on to next line

        longitude.append(f['/' + line   + land_seg_path + 'longitude/'][...,].tolist())

        # Get ground track
        Arrtemp = f['/' + line  + land_seg_path + 'latitude/'][...,]
        temp = np.empty_like(Arrtemp, dtype='a255')
        temp[...] = line
        gt.append(temp)

        segid_beg.append(f['/' + line   + land_seg_path + 'segment_id_beg/'][...,].tolist())
        segid_end.append(f['/' + line   + land_seg_path + 'segment_id_end/'][...,].tolist())
        
        # Canopy fields
        #if get_gedi_rh:
        #    RH_TYPE = 'gedi_rh'
        #else:
        #    RH_TYPE = 'atl03_rh'
        
        if do_30m:

            can_h_met_0.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_25/'][...,].tolist() )
            can_h_met_1.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_30/'][...,].tolist() )
            can_h_met_2.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_40/'][...,].tolist() )
            can_h_met_3.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_50/'][...,].tolist() )
            can_h_met_4.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_60/'][...,].tolist() )
            can_h_met_5.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_70/'][...,].tolist() )
            can_h_met_6.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_75/'][...,].tolist() )
            can_h_met_7.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_80/'][...,].tolist() )
            can_h_met_8.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_90/'][...,].tolist() )
            
            if TEST:
                pass
                #print(len(can_h_met_0), len(can_h_met_1), len(can_h_met_2))
                
            h_max_can.append(f['/' + line   + f'/land_segments/30m_segment/{rh_type}_rh_100/'][...,].tolist())
            h_can.append(f['/' + line       + f'/land_segments/30m_segment/{rh_type}_rh_98/'][...,].tolist())
            h_can_quad.append(f['/' + line  + '/land_segments/30m_segment/h_canopy_quad'][...,].tolist())
            h_can_unc.append(f['/' + line   + '/land_segments/30m_segment/h_canopy_uncertainty'][...,].tolist())
            
            n_ca_ph.append(f['/' + line     + '/land_segments/30m_segment/n_ca_photons/'][...,].tolist())
            n_toc_ph.append(f['/' + line    + '/land_segments/30m_segment/n_toc_photons/'][...,].tolist())
            can_open.append(f['/' + line    + '/land_segments/30m_segment/canopy_openness/'][...,].tolist())
            can_rh_conf.append(f['/' + line + '/land_segments/30m_segment/canopy_rh_conf/'][...,].tolist())
            
            if(atl08_version > 4):
                seg_cover.append(f['/' + line   + '/land_segments/30m_segment/segment_cover/'][...,].tolist())
            else:
                tcc_flg.append(f['/' + line     + '/land_segments/30m_segment/landsat_flag/'][...,].tolist())
                tcc_prc.append(f['/' + line     + '/land_segments/30m_segment/landsat_perc/'][...,].tolist())
        else:
            can_h_met.append(f['/' + line   + '/land_segments/canopy/canopy_h_metrics/'][...,].tolist())
            
            h_max_can.append(f['/' + line   + '/land_segments/canopy/h_max_canopy/'][...,].tolist())
            h_can.append(f['/' + line       + '/land_segments/canopy/h_canopy/'][...,].tolist())
            h_can_quad.append(f['/' + line  + '/land_segments/canopy/h_canopy_quad'][...,].tolist())
            h_can_unc.append(f['/' + line   + '/land_segments/canopy/h_canopy_uncertainty'][...,].tolist())
            
            n_ca_ph.append(f['/' + line     + '/land_segments/canopy/n_ca_photons/'][...,].tolist())
            n_toc_ph.append(f['/' + line    + '/land_segments/canopy/n_toc_photons/'][...,].tolist())
            can_open.append(f['/' + line    + '/land_segments/canopy/canopy_openness/'][...,].tolist())
            can_rh_conf.append(f['/' + line + '/land_segments/canopy/canopy_rh_conf/'][...,].tolist())

            if(atl08_version > 4):
                seg_cover.append(f['/' + line   + '/land_segments/canopy/segment_cover/'][...,].tolist())
            else:
                tcc_flg.append(f['/' + line     + '/land_segments/canopy/landsat_flag/'][...,].tolist())
                tcc_prc.append(f['/' + line     + '/land_segments/canopy/landsat_perc/'][...,].tolist())
      
    
        # Uncertinaty fields
        cloud_flg.append(f['/' + line   + land_seg_path + 'cloud_flag_atm/'][...,].tolist())
        msw_flg.append(f['/' + line     + land_seg_path + 'msw_flag/'][...,].tolist())
        n_seg_ph.append(f['/' + line    + land_seg_path + 'n_seg_ph/'][...,].tolist())
        night_flg.append(f['/' + line   + land_seg_path + 'night_flag/'][...,].tolist())
        seg_landcov.append(f['/' + line + land_seg_path + 'segment_landcover/'][...,].tolist())
        seg_snow.append(f['/' + line    + land_seg_path + 'segment_snowcover/'][...,].tolist())
        seg_water.append(f['/' + line   + land_seg_path + 'segment_watermask/'][...,].tolist())
        sig_vert.append(f['/' + line    + land_seg_path + 'sigma_atlas_land/'][...,].tolist())
        sig_acr.append(f['/' + line     + land_seg_path + 'sigma_across/'][...,].tolist())
        sig_along.append(f['/' + line   + land_seg_path + 'sigma_along/'][...,].tolist())
        sig_h.append(f['/' + line       + land_seg_path + 'sigma_h/'][...,].tolist())
        sig_topo.append(f['/' + line    + land_seg_path + 'sigma_topo/'][...,].tolist())

        # Terrain fields
        if do_30m:
            n_te_ph.append(f['/' + line     + '/land_segments/30m_segment/n_te_photons/'][...,].tolist())
            h_te_best.append(f['/' + line   + '/land_segments/30m_segment/h_te_best_fit/'][...,].tolist())
            h_te_unc.append(f['/' + line    + '/land_segments/30m_segment/h_te_uncertainty/'][...,].tolist())
            ter_slp.append(f['/' + line     + '/land_segments/30m_segment/terrain_slope/'][...,].tolist())
        else:
            n_te_ph.append(f['/' + line     + '/land_segments/terrain/n_te_photons/'][...,].tolist())
            h_te_best.append(f['/' + line   + '/land_segments/terrain/h_te_best_fit/'][...,].tolist())
            h_te_unc.append(f['/' + line    + '/land_segments/terrain/h_te_uncertainty/'][...,].tolist())
            ter_slp.append(f['/' + line     + '/land_segments/terrain/terrain_slope/'][...,].tolist())
            
        snr.append(f['/' + line         + land_seg_path + 'snr/'][...,].tolist())
        sol_az.append(f['/' + line      + land_seg_path + 'solar_azimuth/'][...,].tolist())
        sol_el.append(f['/' + line      + land_seg_path + 'solar_elevation/'][...,].tolist())

        asr.append(f['/' + line         + land_seg_path + 'asr/'][...,].tolist())
        h_dif_ref.append(f['/' + line   + land_seg_path + 'h_dif_ref/'][...,].tolist())
        ter_flg.append(f['/' + line     + land_seg_path + 'terrain_flg/'][...,].tolist())
        ph_rem_flg.append(f['/' + line  + land_seg_path + 'ph_removal_flag/'][...,].tolist())
        dem_rem_flg.append(f['/' + line + land_seg_path + 'dem_removal_flag/'][...,].tolist())
        seg_wmask.append(f['/' + line   + land_seg_path + 'segment_watermask/'][...,].tolist())
        lyr_flg.append(f['/' + line     + land_seg_path + 'layer_flag/'][...,].tolist())
    
    # MW 3/31: Originally a length of 6 was hardcoded into the below calculations because the
    #          assumption was made that 6 lines/lasers worth of data was stored in the arrays. With
    #	       the above changes made to the beginning of the 'for line in lines' loop on 3/31, this
    #          assumption is no longer always true. Adding nLines var to replace range(6) below
    nLines = len(latitude)

    # Be sure at least one of the lasers/lines for the h5 file had data points - MW added block 3/31
    if nLines == 0:
        return None # No usable points in h5 file, can't process
    if TEST:
        pass
        #print("Convert the list of lists into a single list...")
        
    latitude    =np.array([latitude[l][k] for l in range(nLines) for k in range(len(latitude[l]))] )
    longitude   =np.array([longitude[l][k] for l in range(nLines) for k in range(len(longitude[l]))] )

    gt          =np.array([gt[l][k] for l in range(nLines) for k in range(len(gt[l]))] )

    segid_beg   =np.array([segid_beg[l][k] for l in range(nLines) for k in range(len(segid_beg[l]))] )
    segid_end   =np.array([segid_end[l][k] for l in range(nLines) for k in range(len(segid_end[l]))] )

    
    h_max_can   =np.array([h_max_can[l][k] for l in range(nLines) for k in range(len(h_max_can[l]))] )
    h_can       =np.array([h_can[l][k] for l in range(nLines) for k in range(len(h_can[l]))] )
    h_can_unc   =np.array([h_can_unc[l][k] for l in range(nLines) for k in range(len(h_can_unc[l]))] )
    h_can_quad  =np.array([h_can_quad[l][k] for l in range(nLines) for k in range(len(h_can_quad[l]))] )
    
    if do_30m:
        can_h_met_0 = np.array([can_h_met_0[l][k] for l in range(nLines) for k in range(len(can_h_met_0[l]))])
        can_h_met_1 = np.array([can_h_met_1[l][k] for l in range(nLines) for k in range(len(can_h_met_1[l]))])
        can_h_met_2 = np.array([can_h_met_2[l][k] for l in range(nLines) for k in range(len(can_h_met_2[l]))])
        can_h_met_3 = np.array([can_h_met_3[l][k] for l in range(nLines) for k in range(len(can_h_met_3[l]))])
        can_h_met_4 = np.array([can_h_met_4[l][k] for l in range(nLines) for k in range(len(can_h_met_4[l]))])
        can_h_met_5 = np.array([can_h_met_5[l][k] for l in range(nLines) for k in range(len(can_h_met_5[l]))])
        can_h_met_6 = np.array([can_h_met_6[l][k] for l in range(nLines) for k in range(len(can_h_met_6[l]))])
        can_h_met_7 = np.array([can_h_met_7[l][k] for l in range(nLines) for k in range(len(can_h_met_7[l]))])
        can_h_met_8 = np.array([can_h_met_8[l][k] for l in range(nLines) for k in range(len(can_h_met_8[l]))])
    else:
        can_h_met = np.array([can_h_met[l][k] for l in range(nLines) for k in range(len(can_h_met[l]))])
        
    n_ca_ph     =np.array([n_ca_ph[l][k] for l in range(nLines) for k in range(len(n_ca_ph[l]))] )
    n_toc_ph    =np.array([n_toc_ph[l][k] for l in range(nLines) for k in range(len(n_toc_ph[l]))] )
    can_open    =np.array([can_open[l][k] for l in range(nLines) for k in range(len(can_open[l]))] )
    can_rh_conf =np.array([can_rh_conf[l][k] for l in range(nLines) for k in range(len(can_rh_conf[l]))] )
    if(atl08_version > 4):
        seg_cover    =np.array([seg_cover[l][k] for l in range(nLines) for k in range(len(seg_cover[l]))] )
    else:
        tcc_flg     =np.array([tcc_flg[l][k] for l in range(nLines) for k in range(len(tcc_flg[l]))] )
        tcc_prc     =np.array([tcc_prc[l][k] for l in range(nLines) for k in range(len(tcc_prc[l]))] )

    cloud_flg   =np.array([cloud_flg[l][k] for l in range(nLines) for k in range(len(cloud_flg[l]))] )
    msw_flg     =np.array([msw_flg[l][k] for l in range(nLines) for k in range(len(msw_flg[l]))] )
    n_seg_ph    =np.array([n_seg_ph[l][k] for l in range(nLines) for k in range(len(n_seg_ph[l]))] )
    night_flg    =np.array([night_flg[l][k] for l in range(nLines) for k in range(len(night_flg[l]))] )
    seg_landcov =np.array([seg_landcov[l][k] for l in range(nLines) for k in range(len(seg_landcov[l]))] )
    seg_snow    =np.array([seg_snow[l][k] for l in range(nLines) for k in range(len(seg_snow[l]))] )
    seg_water   =np.array([seg_water[l][k] for l in range(nLines) for k in range(len(seg_water[l]))] )
    sig_vert    =np.array([sig_vert[l][k] for l in range(nLines) for k in range(len(sig_vert[l]))] )
    sig_acr     =np.array([sig_acr[l][k] for l in range(nLines) for k in range(len(sig_acr[l]))] )
    sig_along   =np.array([sig_along[l][k] for l in range(nLines) for k in range(len(sig_along[l]))] )
    sig_h       =np.array([sig_h[l][k] for l in range(nLines) for k in range(len(sig_h[l]))] )
    sig_topo    =np.array([sig_topo[l][k] for l in range(nLines) for k in range(len(sig_topo[l]))] )

    n_te_ph     =np.array([n_te_ph[l][k] for l in range(nLines) for k in range(len(n_te_ph[l]))] )
    h_te_best   =np.array([h_te_best[l][k] for l in range(nLines) for k in range(len(h_te_best[l]))] )
    h_te_unc    =np.array([h_te_unc[l][k] for l in range(nLines) for k in range(len(h_te_unc[l]))] )
    ter_slp     =np.array([ter_slp[l][k] for l in range(nLines) for k in range(len(ter_slp[l]))] )
    snr         =np.array([snr[l][k] for l in range(nLines) for k in range(len(snr[l]))] )
    sol_az      =np.array([sol_az[l][k] for l in range(nLines) for k in range(len(sol_az[l]))] )
    sol_el      =np.array([sol_el[l][k] for l in range(nLines) for k in range(len(sol_el[l]))] )

    asr 		=np.array([asr[l][k] for l in range(nLines) for k in range(len(asr[l]))] )
    h_dif_ref 	=np.array([h_dif_ref[l][k] for l in range(nLines) for k in range(len(h_dif_ref[l]))] )
    ter_flg 	=np.array([ter_flg[l][k] for l in range(nLines) for k in range(len(ter_flg[l]))] )
    ph_rem_flg	=np.array([ph_rem_flg[l][k] for l in range(nLines) for k in range(len(ph_rem_flg[l]))] )
    dem_rem_flg =np.array([dem_rem_flg[l][k] for l in range(nLines) for k in range(len(dem_rem_flg[l]))] )
    seg_wmask	=np.array([seg_wmask[l][k] for l in range(nLines) for k in range(len(seg_wmask[l]))] )
    lyr_flg		=np.array([lyr_flg[l][k] for l in range(nLines) for k in range(len(lyr_flg[l]))] )

    # Handle nodata
    val_invalid = np.finfo('float32').max
    val_nan = np.nan
    val_nodata_src = np.max(h_can)
    print("Find src nodata value using max of h_can: \t{}".format(val_nodata_src))
    
    if TEST:
        
        # Testing with 'h_can'
        
        print("\n\tNan used for 30m version, not for 100m version.")
        print("\t\tCheck max of h_can...")
        print("\t\tnp.nanmax: \t{}".format(np.nanmax(h_can)) )
        print("\t\tnp.max: \t{}".format(np.max(h_can)) )
        
           
        
        print('[BEFORE] # of nan ATL08 obs of h_can: \t{}'.format(len( h_can[np.isnan(h_can) ] )))
        h_can = np.array([val_invalid if math.isnan(x) else x for x in h_can])
        print("Set h_can max to float32 max; np.max: \t {}".format(np.max(h_can)))
        print('[AFTER] # of nan ATL08 obs of h_can: \t{}'.format(len( h_can[np.isnan(h_can) ] )))
        
        print("\nData type: \t{}".format(h_can.dtype))
        print("# of nan ATL08 obs of  h_can: \t{}".format( np.count_nonzero(np.isnan(h_can)) ))
        print('# of invalid ATL08 obs of h_can: \t{}'.format(len( h_can[h_can == val_invalid ] )))
        
        print('# of ATL08 obs: \t\t{}'.format(len(latitude)))
        print('# of ATL08 obs (can pho.>0): \t{}'.format(len(n_ca_ph[n_ca_ph>0])))
        print('# of ATL08 obs (toc pho.>0): \t{}'.format(len(n_toc_ph[n_toc_ph>0])))
        print('# of ATL08 obs (h_can>0): \t{}'.format(len(h_can[h_can>0])))

    # Get approx path center Lat
    #CenterLat = latitude[len(latitude)/2]
    CenterLat = latitude[int(len(latitude)/2)]

    if False:
        # Calc args.resolution in degrees
        ellipse = [6378137.0, 6356752.314245]
        radlat = np.deg2rad(CenterLat)
        Rsq = (ellipse[0]*np.cos(radlat))**2+(ellipse[1]*np.sin(radlat))**2
        Mlat = (ellipse[0]*ellipse[1])**2/(Rsq**1.5)
        Nlon = ellipse[0]**2/np.sqrt(Rsq)
        pixelSpacingInDegreeX = float(args.resolution) / (np.pi/180*np.cos(radlat)*Nlon)
        pixelSpacingInDegreeY = float(args.resolution) / (np.pi/180*Mlat)
        print('Raster X (' + str(args.resolution) + ' m) Resolution at ' + str(CenterLat) + ' degrees N = ' + str(pixelSpacingInDegreeX))
        print('Raster Y (' + str(args.resolution) + ' m) Resolution at ' + str(CenterLat) + ' degrees N = ' + str(pixelSpacingInDegreeY))

    # Create a handy ID label for each point
    fid = np.arange(1, len(h_max_can)+1, 1)

    if TEST:
        print("\nSet up a dataframe dictionary...")
 
    dict_orb_gt_seg = {
                    'fid'       :fid,
                    'lon'       :longitude,
                    'lat'       :latitude,

                    #'yr'        :np.full(longitude.shape, yr[0]),
                    #'m'         :np.full(longitude.shape, m[0]),
                    #'d'         :np.full(longitude.shape, d[0]),
                    'dt'        :np.full(longitude.shape, dt[0]),
                    'orb_orient':np.full(longitude.shape, orb_orient[0]),
                    'orb_num'   :np.full(longitude.shape, orb_num[0]),
                    'rgt'       :np.full(longitude.shape, rgt[0]),
                    'gt'        :gt,

                    'segid_beg' :segid_beg,
                    'segid_end' :segid_end
    }
    if do_30m:
        dict_rh_metrics = {
                        'h_max_can' :h_max_can,
                        'h_can'     :h_can,
                        'h_can_quad':h_can_quad,
                        'h_can_unc' :h_can_unc,

                        'rh25'      :can_h_met_0,
                        'rh30'      :can_h_met_1,
                        'rh40'      :can_h_met_2,
                        'rh50'      :can_h_met_3,
                        'rh60'      :can_h_met_4,
                        'rh70'      :can_h_met_5,
                        'rh75'      :can_h_met_6,
                        'rh80'      :can_h_met_7,
                        'rh90'      :can_h_met_8     
        }
    else:
        dict_rh_metrics = {
                        'h_max_can' :h_max_can,
                        'h_can'     :h_can,
                        'h_can_quad':h_can_quad,
                        'h_can_unc' :h_can_unc,

                        'rh25'      :can_h_met[:,0],
                        'rh50'      :can_h_met[:,1],
                        'rh60'      :can_h_met[:,2],
                        'rh70'      :can_h_met[:,3],
                        'rh75'      :can_h_met[:,4],
                        'rh80'      :can_h_met[:,5],
                        'rh85'      :can_h_met[:,6],
                        'rh90'      :can_h_met[:,7],
                        'rh95'      :can_h_met[:,8]
        }
    dict_misc_fields = {
                    'n_ca_ph'   :n_ca_ph,
                    'n_toc_ph'  :n_toc_ph,
                    'can_open'  :can_open,
                    'can_rh_conf':can_rh_conf,

                    'cloud_flg' :cloud_flg,
                    'msw_flg'   :msw_flg,
                    'n_seg_ph'  :n_seg_ph,
                    'night_flg' :night_flg,
                    'seg_landcov':seg_landcov,
                    'seg_snow'  :seg_snow,
                    'seg_water' :seg_water,
                    'sig_vert'  :sig_vert,
                    'sig_acr'   :sig_acr,
                    'sig_along' :sig_along,
                    'sig_h'     :sig_h,
                    'sig_topo'  :sig_topo,

                    'n_te_ph'   :n_te_ph,
                    'h_te_best' :h_te_best,
                    'h_te_unc'  :h_te_unc,
                    'ter_slp'   :ter_slp,
                    'snr'       :snr,
                    'sol_az'    :sol_az,
                    'sol_el'    :sol_el,

                    'asr'       :asr,
                    'h_dif_ref' :h_dif_ref,
                    'ter_flg'   :ter_flg,
                    'ph_rem_flg':ph_rem_flg,
                    'dem_rem_flg':dem_rem_flg,
                    'seg_wmask' :seg_wmask,
                    'lyr_flg'   :lyr_flg
    }

    if(atl08_version > 4):
        dict_version_dep_fields = {
                                'seg_cover'  :seg_cover
        }
    else:
        dict_version_dep_fields = {
                                'tcc_flg'   :tcc_flg,
                                'tcc_prc'   :tcc_prc
                                    }
        
    print("\nBuilding pandas dataframe...")
    out = pd.DataFrame(rec_merge1(dict_orb_gt_seg, rec_merge1(dict_rh_metrics, rec_merge1(dict_misc_fields, dict_version_dep_fields))))
    
    print("Setting pandas df nodata values to np.nan for some basic eval.")
    out = out.replace(val_nodata_src, np.nan)
   
    print('# of ATL08 obs: \t\t{}'.format(len(out.lat[out.lat.notnull()])))
    print('# of ATL08 obs (can pho.>=0): \t{}'.format(len(out.n_ca_ph[
                                                                    (out.h_can.notnull() ) & 
                                                                    (out.n_ca_ph >= 0) 
                                                                ])))
    print('# of ATL08 obs (toc pho.>=0): \t{}'.format(len(out.n_toc_ph[
                                                                    (out.h_can.notnull() ) & 
                                                                    (out.n_toc_ph >= 0) 
                                                                ])))
    print('# of ATL08 obs (h_can>=0): \t{}'.format(len(out.h_can[
                                                                (out.h_can.notnull() ) & 
                                                                (out.h_can >= 0) 
                                                               ])))
    print('# of ATL08 obs (h_can<0): \t{}'.format(len(out.h_can[
                                                                (out.h_can.notnull() ) & 
                                                                (out.h_can < 0) 
                                                               ])))
    if args.set_nodata_nan:
        val_nodata_out = val_nan
    else:
        val_nodata_out = val_invalid
    print("Setting out pandas df nodata values: \t{}".format(val_nodata_out))
    out = out.replace(np.nan, val_nodata_out)
    
    if args.set_flag_names:
        # Set flag names
        if(atl08_version > 4):
            class_values = [ 0, 111, 113, 112, 114, 115, 116, 121, 123, 122, 124, 125, 126, 20, 30, 90, 100, 60, 40, 50, 70, 80, 200] 
            class_names = ['No data','Closed forest\nevergreen needle','Closed forest\ndeciduous needle','Closed forest\nevergreen_broad','Closed forest\ndeciduous broad','Closed forest\nmixed', 'Closed forest\nunknown','Open forest\nevergreen needle',
                'Open forest deciduous needle','Open forest evergreen_broad','Open forest deciduous_broad','Open forest mixed', 'Open forest unknown', 'Shrubs','Herbaceous', 'Herbaceous\nwetleand','Moss/lichen', 'Bare/sparse','Cultivated/managed',
                'Urban/built', 'Snow/ice','Permanent\nwater', 'Open sea']
            out['seg_landcov'] = out['seg_landcov'].map(dict(zip(class_values, class_names)))
        else:
            out['seg_landcov'] = out['seg_landcov'].map({0: "water", 1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest", \
                                                         3: "deciduous needleleaf forest", 4: "deciduous broadleaf forest", \
                                                         5: "mixed forest", 6: "closed shrublands", 7: "open shrublands", \
                                                         8: "woody savannas", 9: "savannas", 10: "grasslands", 11: "permanent wetlands", \
                                                         12: "croplands", 13: "urban-built", 14: "croplands-natural mosaic", \
                                                         15: "permanent snow-ice", 16: "barren"})
        out['seg_snow'] = out['seg_snow'].map({0: "ice free water", 1: "snow free land", 2: "snow", 3: "ice"})
        out['cloud_flg'] = out['cloud_flg'].map({0: "High conf. clear skies", 1: "Medium conf. clear skies", 2: "Low conf. clear skies", \
                                                 3: "Low conf. cloudy skies", 4: "Medium conf. cloudy skies", 5: "High conf. cloudy skies"})
        out['night_flg'] = out['night_flg'].map({0: "day", 1: "night"})
        #out['tcc_flg'] = out['tcc_flg'].map({0: "=<5%", 1: ">5%"})
                                         
    ## Bin tcc values                                     
    #tcc_bins = [0,10,20,30,40,50,60,70,80,90,100]
    #out['tcc_bin'] = pd.cut(out['tcc_prc'], bins=tcc_bins, labels=tcc_bins[1:])

    # Add granule name to table
    out['granule_name'] = granule_fname
    # Add rh metric type to table
    out['rh_type'] = rh_type
    
    if filter_qual:

        print('Quality Filtering: \t\t[ON]')

        import FilterUtils

        # These filters are customized for boreal
        '''out = FilterUtils.prep_filter_atl08_qual(out)
        out = FilterUtils.filter_atl08_qual_v2(out, SUBSET_COLS=True, DO_PREP=False,
                                                   subset_cols_list=['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can','h_max_can','h_can_quad','h_can_unc',
                                                                     'h_te_best','h_te_unc', 'granule_name','can_rh_conf', 'h_dif_ref',
                                                                     'seg_landcov','seg_cover','night_flg','seg_water','sol_el','asr','ter_slp', 'ter_flg','y','m','d'], 
                                                   filt_cols=['h_can','h_dif_ref','m','msw_flg','beam_type','seg_snow','sig_topo'], 
                                                   thresh_h_can=100, thresh_h_dif=25, thresh_sig_topo=2.5, month_min=args.minmonth, month_max=args.maxmonth)
                                                   '''
        print('Apply the aggressive land-cover based (v3) filters updated in Jan/Feb 2022')
        out = FilterUtils.filter_atl08_qual_v3(out, SUBSET_COLS=True, DO_PREP=True,
                                              subset_cols_list=['rh25','rh50','rh60','rh70','rh75','rh80','rh90','h_can','h_max_can',
                                                                     'h_te_best','granule_name', 'rh_type',
                                                                     'seg_landcov','seg_cover','sol_el','y','m','doy'], 
                                                   filt_cols=['h_can','h_dif_ref','m','msw_flg','beam_type','seg_snow','sig_topo'], 
                                                   list_lc_h_can_thresh=args.list_lc_h_can_thresh,
                                                   thresh_h_can=100, thresh_h_dif=25, thresh_sig_topo=2.5, month_min=args.minmonth, month_max=args.maxmonth)

    else:
        print('Quality Filtering: \t[OFF] (do downstream)')

    if args.filter_geo:
        print('Geographic Filtering: \t[ON] xmin = {}, xmax = {}, ymin = {}, ymax = {}'.format(args.minlon, args.maxlon, args.minlat, args.maxlat))        
        # These filters are customized for boreal 
        out = out[ (out['lon']     >= args.minlon) & 
                   (out['lon']     <= args.maxlon) & 
                   (out['lat']     >= args.minlat) & 
                   (out['lat']     <= args.maxlat)
                 ]
    else:
        print('Geographic Filtering: \t[OFF] (do downstream)')

    if out.empty:
        print('File is empty.')
    else:
        if args.output_dataframe:
            print(f'Returning output dataframe of shape: {out.shape}')
            return out
        else:
            # Write out to a csv
            out_csv_fn = os.path.join(outbase + fn_tail)
            print('Creating CSV: \t\t{}'.format(out_csv_fn))
            out.to_csv(out_csv_fn,index=False, encoding="utf-8-sig")


def main():
    print("\nWritten by:\n\tNathan Thomas\t| @Nmt28\n\tPaul Montesano\t| paul.m.montesano@nasa.gov\n")
                                         
    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, other):
            return self.start <= other <= self.end

        def __contains__(self, item):
            return self.__eq__(item)

        def __iter__(self):
            yield self

        def __str__(self):
            return '[{0},{1}]'.format(self.start, self.end)                                           

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Specify the input ICESAT H5 file")
    parser.add_argument("-r", "--resolution", type=str, default='100', help="Specify the output raster resolution (m)")
    parser.add_argument("-o", "--out_dir", type=str, help="Specify the output directory (optional)")
    parser.add_argument("-v", "--out_var", type=str, default='h_max_can', help="A selected variable to rasterize")
    #parser.add_argument("-prj", "--out_epsg", type=str, default='102001', help="Out raster prj (default: Canada Albers Equal Area)")
    parser.add_argument("--max_h_can" , type=float, choices=[Range(0.0, 100.0)], default=30.0, help="Max value of h_can to include")
    parser.add_argument("--min_n_toc_ph" , type=int, default=1, help="Min number of top of canopy classified photons required for shot to be output")
    parser.add_argument("--minlon" , type=float, choices=[Range(-180.0, 180.0)], default=-180.0, help="Min longitude of ATL08 shots for output to include") 
    parser.add_argument("--maxlon" , type=float, choices=[Range(-180.0, 180.0)], default=180.0, help="Max longitude of ATL08 shots for output to include")
    parser.add_argument("--minlat" , type=float, choices=[Range(-90.0, 90.0)], default=30.0, help="Min latitude of ATL08 shots for output to include") 
    parser.add_argument("--maxlat" , type=float, choices=[Range(-90.0, 90.0)], default=80.0, help="Max latitude of ATL08 shots for output to include")
    parser.add_argument("--minmonth" , type=int, choices=[Range(1, 12)], default=6, help="Min month of ATL08 shots for output to include")
    parser.add_argument("--maxmonth" , type=int, choices=[Range(1, 12)], default=9, help="Max month of ATL08 shots for output to include")
    parser.add_argument("-t", "--rh_type", choices=['gedi', 'atl03'], nargs="?", type=str, default='ATL08', const='ATL08', help="Specify the RH metric calc type; GEDI calc includes ground photons, ATL03 does not")
    parser.add_argument("-l", "--seg_length", choices=[30, 100], nargs="?", type=int, default=30, const=30, help='Choose 30m or 100m ATL08 extraction')
    parser.add_argument("--list_lc_h_can_thresh", nargs="+", type=int, default=[0, 60, 60, 60, 60, 60, 60, 50, 50, 50, 50, 50, 50, 20, 10, 10, 5, 5, 0, 0, 0, 0, 0], help="A list of land-cover specific thresholds for h_can")
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false', help='Turn overwrite off (To help complete big runs that were interrupted)')
    parser.set_defaults(overwrite=True)
    parser.add_argument('--no-filter-qual', dest='filter_qual', action='store_false', help='Turn off quality filtering (To control filtering downstream)')
    parser.set_defaults(filter_qual=True)
    parser.add_argument('--no-filter-geo', dest='filter_geo', action='store_false', help='Turn off geographic filtering (To control filtering downstream)')
    parser.set_defaults(filter_geo=True)
    #parser.add_argument('--do_30m', dest='do_30m', action='store_true', help='Turn on 30m ATL08 extraction')
    #parser.set_defaults(do_30m=False)
    #parser.add_argument('--get_gedi_rh', dest='get_gedi_rh', action='store_true', help='Get rh metrics from the GEDI version (which considers ground photons)')
    #parser.set_defaults(get_gedi_rh=False)
    parser.add_argument('--output_dataframe', dest='output_dataframe', action='store_true', help='Output a pandas dataframe instead of a csv')
    parser.set_defaults(output_dataframe=False)
    parser.add_argument('--set_flag_names', dest='set_flag_names', action='store_true', help='Set the flag values to meaningful flag names')
    parser.set_defaults(set_flag_names=False)
    parser.add_argument('--set_nodata_nan', dest='set_nodata_nan', action='store_true', help='Set output nodata to nan')
    parser.set_defaults(set_nodata_nan=False)
    parser.add_argument('--TEST', dest='TEST', action='store_true', help='Turn on testing')
    parser.set_defaults(TEST=False)
    

    args = parser.parse_args()


    if str(args.input).endswith('.h5'):
        pass
    else:
        print("INPUT ICESAT2 FILE MUST END '.H5'")
        os._exit(1)
    if args.output == None:
        print("\n OUTPUT DIR IS NOT SPECIFIED (OPTIONAL). OUTPUT WILL BE PLACED IN THE SAME LOCATION AS INPUT H5 \n\n")
    else:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        else:
            pass
    if args.resolution == None:
        print("SPECIFY OUTPUT RASTER RESOLUTION IN METERS'")
        os._exit(1)
    else:
        pass

    if args.filter_geo:
        print("Min lat: {}".format(args.minlat))
        print("Max lat: {}".format(args.maxlat))
        print("Min lon: {}".format(args.minlon))
        print("Max lon: {}".format(args.maxlon))

    print(f'Month range: {args.minmonth}-{args.maxmonth}')

    extract_atl08(args)


if __name__ == "__main__":
    main()