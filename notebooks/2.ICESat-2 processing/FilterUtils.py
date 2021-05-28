import pandas as pd
import geopandas as gpd

def above_filter_atl08(input_fn=None, out_cols_list=[], thresh_h_can=None, thresh_h_dif=None, month_min=None, month_mas=None):
    
    if not out_cols_list:
        print("above_filter_atl08: Must supply a list of strings matching ATL08 column names returned from the input EPT")
        os._exit(1) 
    elif thresh_h_can=None:
        print("above_filter_atl08: Must supply a threshold for h_can")
        os._exit(1)    
    elif thresh_h_dif=None:
        print("above_filter_atl08: Must supply a threshold for h_dif_ref")
        os._exit(1)
    elif month_min=None or month_max=None:
        print("above_filter_atl08: Must supply a month_min and month_max")
        os._exit(1)  
        
    if input_fn is not None:
        if input_fn.endswith('geojson')
            atl08_df = gpd.read(input_fn)
        elif input_fn.endswith('csv'):
            atl08_df = pd.read_csv(input_fn)
        else:
            print("Input filename must be a CSV or GEOJSON")
            os._exit(1)
                
    # Filtering
    #
    # Filter list (keep):
    #   h_ref_diff < THRESH_h_dif
    #   h_can < THRESH_h_can
    #   no LC forest masking: only forest LC classes no good b/c trees outside of forest aer of interest (woodlands, etc)
    #   msw = 0
    #   night better (but might exclude too much good summer data in the high northern lats)
    #   strong beam
    #   summer (june - mid sept)
    #   seg_snow == 'snow free land'
        
    print("\nFiltering for quality:\n\tfor clear skies + strong beam + snow free land,\n\th_can < {},\n\televation diff from ref < {},\n\tmonths {}-{}".format(thresh_h_can, thresh_h_dif, month_min, month_max))
    atl08_df_filt =  atl08_df[
                                (atl08_df.h_can < THRESH_h_can) &
                                (atl08_df.h_dif_ref < THRESH_h_dif) &
                                (atl08_df.m >= MONTH_MIN ) & (atl08_pdf.m <= MONTH_MAX) &
                                # Hard coded quality flags for ABoVE AGB
                                (atl08_df.msw_flg == 0) &
                                #(atl08_df.night_flg == 'night') & # might exclude too much good summer data in the high northern lats
                                (atl08_df.beam_type == 'Strong') & 
                                (atl08_df.seg_snow == 'snow free land')
                    ]
    print(f"Before filtering: {atl08_df.shape[0]} observations in the input dataframe.")
    print(f"After filtering: {atl08_df_filt.shape[0]} observations in the output dataframe.")
    print("Returning a pandas data frame of filtered observations for columns: {}".format(in_cols_list))
    return(atl08_df_filt)