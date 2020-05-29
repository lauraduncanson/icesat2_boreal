#! /bin/bash
#
# This will filter an ICESat-2 ATL08 CSV
#
# run like this from terminal or bash script:
# Rscript ~/code/icesat2/merge_atl08.R --indir="$NOBACKUP/userfs02/data/icesat2/atl08/v3/csv" --csv_search_str="003_01_filt_45_90_-180_-13"

suppressMessages(library(rgdal))
suppressMessages(library(raster))
suppressMessages(library(stringr))
suppressMessages(library(sf))
suppressMessages(library(fs))
suppressMessages(library(tidyverse))
suppressMessages(library(optparse))
suppressMessages(library(tools))
suppressMessages(library(plyr))

option_list = list(
  make_option(c("--atl08_csv_fn"),    type="character", default=NULL, help="Input ATL08 csv filename from extract_atl08.py [default= %default]", metavar="character"),
  make_option(c("--outdir"),          type="character", default=NULL, help="Output dir of filtered csv file [default= %default]"),
  make_option(c("--filt_minlat"),     type="numeric", default=45, help="Filter for min lat [default= %default]"),
  make_option(c("--filt_maxlat"),     type="numeric", default=90, help="Filter for max lat [default= %default]"),
  make_option(c("--filt_minlon"),     type="numeric", default=NULL, help="Filter for min lon [default= %default]"),
  make_option(c("--filt_maxlon"),     type="numeric", default=NULL, help="Filter for max lon [default= %default]"),
  make_option(c("--filt_max_h_can"),  type="numeric", default=30, help="Filter for canopy height threshold of h_can metric [default= %default]")

);

main <- function(){
  
  #''' Filter ICESat-2 ATL08 CSV
  #       return a filtered CSV
  #'''
  print(as.list(match.call()))
  opt_parser = OptionParser(option_list=option_list);
  opt = parse_args(opt_parser);
  
  if(is.null(opt$atl08_csv_fn)){
    print_help(opt_parser)
    stop("Must provide an input CSV filename", call.=FALSE)
  }
  if(is.null(opt$filt_minlon)){
    print_help(opt_parser)
    stop("Must provide a min lon", call.=FALSE)
  }
  if(is.null(opt$filt_maxlon)){
    print_help(opt_parser)
    stop("Must provide a max lon", call.=FALSE)
  }

  if(is.null(opt$outdir)){
    print("Output dir. same as input file")
    outdir = dirname(opt$atl08_csv_fn)
  }else{
    outdir = opt$outdir
  }

  atl08_csv = read.csv(opt$atl08_csv_fn)
  atl08_filt_csv_fn = path(outdir, paste0(tools::file_path_sans_ext(basename(opt$atl08_csv_fn)), "_filt_", opt$filt_minlat, "_", opt$filt_maxlat, "_", opt$filt_minlon,"_",opt$filt_maxlon,".csv"))
  print(paste0("Output filtered csv: ", atl08_filt_csv_fn))
  
    #########################
    # FILTER THE ATL08 
    #
    # Get a filtered data 

    df_filt = atl08_csv %>%
    dplyr::filter(h_can     <= opt$filt_max_h_can)%>%
    dplyr::filter(n_toc_ph  > 0)%>%
    dplyr::filter(h_te_unc  != max(h_te_unc))%>%
    dplyr::filter(ter_slp   != max(ter_slp))%>%
    dplyr::filter(lon >= opt$filt_minlon & lon <= opt$filt_maxlon & lat >= opt$filt_minlat & lat <= opt$filt_maxlat)

    # Old filters
    #dplyr::filter(!can_open == max(df$can_open))%>%
    #dplyr::filter(can_open < 12)%>%
    #dplyr::filter(lat > minlat_boreal)#%>%
    #dplyr::mapvalues(seg_snow,  from = c(0,1,2,3), to=c("ice free water", "snow free land","snow", "ice"))%>%
    #dplyr::mapvalues(night_flg, from = c(0,1), to=c("day","night"))%>%
    #dplyr::mapvalues(tcc_flg, from = c(0,1), to=c("= 5%","> 5%"))
    #dplyr::filter(cloud_flg == 0)%>%
    #dplyr::filter(seg_snow == 0)
    #dplyr::filter(night_flg == 1)
    
    # Assign names to categorical variables that can be used to subset data
    df_filt$cloud_flg = plyr::mapvalues(factor(df_filt$cloud_flg), from = c(0,1,2,3,4,5), 
                             to=c("High conf. clear skies","Medium conf. clear skies", "Low conf. clear skies", "Low conf. cloudy skies", "Medium conf. cloudy skies", "High conf. cloudy skies"))

    df_filt$seg_snow  = plyr::mapvalues(factor(df_filt$seg_snow), from = c(0,1,2,3), to=c("ice free water", "snow free land","snow", "ice"))
    df_filt$night_flg = plyr::mapvalues(factor(df_filt$night_flg), from = c(0,1), to=c("day","night"))
    df_filt$tcc_flg   = plyr::mapvalues(factor(df_filt$tcc_flg), from = c(0,1), to=c("= 5%","> 5%"))
  
    # Make treecover classes
    cc_bin=10
    df_filt$tcc_bin = cut(df_filt$tcc_prc, seq(0,100,cc_bin), labels= as.character(seq(cc_bin,100,cc_bin)) )
  
    names(df_filt)[grep("open",names(df_filt))] <- 'can_open'
  
    # Write out a massive CSV of all ATL08 points from H5 files youve found
    write.csv(df_filt , file = atl08_filt_csv_fn, row.names=FALSE)
}
main()