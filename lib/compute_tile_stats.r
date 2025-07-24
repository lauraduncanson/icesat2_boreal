library(aws.s3)
library(terra)
library(tidyverse)
library(dplyr)

library(aws.s3)
library(tidyverse)
library(dplyr)

library(fs)
library(sf)

#nohup Rscript compute_tile_stats.r ...

get_global_stats_df <- function(r_band){
    #'''Using global(), return a dataframe with calcs of basic stats for a raster band
    #'''
    return(data.frame(cbind(
                    global(r_band, c("sum","min", "mean", "max", "sd"), na.rm=TRUE),
                    global(r_band, quantile, probs=c(0.02,0.25,0.50,0.75,0.98), na.rm=TRUE)
                              )
                   , row.names=NULL)
          )
    }

# Define a function that returns multiple statistics
multi_stats <- function(x, na.rm=TRUE) {
  c(sum = sum(x, na.rm=na.rm),
    min = min(x, na.rm=na.rm),
    mean = mean(x, na.rm=na.rm),
    max = max(x, na.rm=na.rm),
    sd = sd(x, na.rm=na.rm),
    X2 = quantile(x, 0.02, na.rm=na.rm),
    X25 = quantile(x, 0.25, na.rm=na.rm),
    X50 = quantile(x, 0.50, na.rm=na.rm),
    X75 = quantile(x, 0.75, na.rm=na.rm),
    X98 = quantile(x, 0.98, na.rm=na.rm)
   )
}

make_north_bbox_projected <- function(to_crs){

    # Step 1: Define your bounding box (lat/lon coordinates)
    # Format: c(xmin, ymin, xmax, ymax) or c(lon_min, lat_min, lon_max, lat_max)
    bbox_coords <- c(-179.9, 51.6, 179.9, 80) 
    
    bbox_polygon <- st_as_sfc(st_bbox(c(xmin = bbox_coords[1], 
                                   ymin = bbox_coords[2], 
                                   xmax = bbox_coords[3], 
                                   ymax = bbox_coords[4]), 
                                 crs = st_crs(4326)))
    
    # Convert to sf object (adds attributes)
    bbox_sf <- st_sf(id = 1, geometry = bbox_polygon)

    # Reproject the polygon to match raster CRS
    bbox_projected <- st_transform(bbox_sf, to_crs)
    
    return(bbox_projected)
}

prep_boreal_poly <- function(boreal_polygon_file, raster){
    
    if(endsWith(boreal_polygon_file, 'wwf_circumboreal_Dissolve.geojson')){
        boreal_poly = vect(st_transform(st_read(boreal_polygon_file), crs(raster)))
        #boreal_poly = vect(st_transform(st_read(boreal_polygon_file)))
    }
    if(endsWith(boreal_polygon_file, 'wwf_circumboreal_Dissolve_reprj.geojson')){
        boreal_poly <- vect(boreal_polygon_file)
        crs(boreal_poly) <- crs(raster)
    }

    return(boreal_poly)
}

crop_raster_to_north <- function(r){

    # Returns a raster cropped to bbox in lat lon, w/o reprojecting raster
    # if not in bbox, raster is same as orig
    
    bbox_north_coords <- c(-179.999999, 51.6, 179.999999, 80)
    
    raster_extent <- ext(r)
    raster_extent_sf <- st_as_sf(st_as_sfc(st_bbox(c(xmin = xmin(raster_extent), 
                                   ymin = ymin(raster_extent), 
                                   xmax = xmax(raster_extent), 
                                   ymax = ymax(raster_extent) ) )),  crs = crs(r))
    
    #footprint_4326 <- st_transform(extent_sf, crs = 4326)

    raster_extent_vect <- vect(raster_extent_sf) #as.polygons(r[[1]], aggregate = TRUE)
    raster_extent_vect_4326 <- project(raster_extent_vect, "EPSG:4326")
    print(ext(raster_extent_vect_4326))
    
    # Create bbox as SpatVector
    bbox_north_matrix <- matrix(c(bbox_north_coords[1], bbox_north_coords[2],
                           bbox_north_coords[3], bbox_north_coords[2],
                           bbox_north_coords[3], bbox_north_coords[4],
                           bbox_north_coords[1], bbox_north_coords[4],
                           bbox_north_coords[1], bbox_north_coords[2]),
                         ncol = 2, byrow = TRUE)
    
    bbox_north_vect_4326 <- vect(bbox_north_matrix, type = "polygons", crs = "EPSG:4326")
    bbox_north_vect_crs_r <- project(bbox_north_vect_4326, crs(r))

    # Check for intersection
    #intersects <- !is.null(terra::intersect(raster_extent_vect_4326, bbox_north_vect_4326))
    intersects = terra::intersect(raster_extent_vect_4326, bbox_north_vect_4326)
    #return(intersects)
    # this doesnt work
    #intersects <- terra::relate(r, bbox_north_vect_crs_r, relation = "intersects")
        
    if(length(intersects)>0){
        DATA_WAS_CROPPED = TRUE
        # Intersect using terra
        clipped_r_extent_vect_4326 <- intersect(raster_extent_vect_4326, bbox_north_vect_4326)
        
        # Convert back from 4326 to crs(r)
        clipped_r_extent_sf_4326 <- st_as_sf(clipped_r_extent_vect_4326)
        clipped_r_extent_sf = st_transform(clipped_r_extent_sf_4326, crs(r))
        # Mask and crop raster 
        
        masked_r <- terra::mask(r, vect(clipped_r_extent_sf))
        cropped_r = terra::crop(masked_r, vect(clipped_r_extent_sf))
    }else{
        DATA_WAS_CROPPED = FALSE
        cat('Tile has no pixels in NORTH\n')
        cropped_r = r
        rm(r)
        }
    return(list(cropped_r, DATA_WAS_CROPPED))
    }

get_extract_stats_df <- function(raster, polygons){
    #'''Using extract(), return a dataframe with calcs of basic stats for a raster band
    #'''
    return(data.frame(
                terra::extract(raster, polygons, fun=multi_stats, ID=FALSE, weights=FALSE, exact=FALSE) #result <- extract(raster, polygons, fun=multi_stats)
                   , row.names=NULL)
          )
    }

do_tile_band_stats_df <- function(r, tile_num, 
                                  bnames=c('agbd_mean','agbd_sd'),
  boreal_polygon_file='/projects/shared-buckets/nathanmthomas/analyze_agb/input_zones/wwf_circumboreal_Dissolve_reprj.geojson',
  #boreal_polygon_file='/projects/shared-buckets/montesano/databank/arc/wwf_circumboreal_Dissolve.geojson',
                                  COMPUTE_VAR=TRUE
                                 ){
    
    #'''Return a tile-level raster csv of combined dataframes for each band of a raster for a full tile, its boreal compoonent, and its northern component 
    #'''

    names(r) = bnames
    
    if(COMPUTE_VAR){
        bnames = c(bnames, 'agbd_var') 
        }
    bnum_list = seq_along(bnames)
    
    # # To calculate just the boreal total
    # boreal_poly <- vect(boreal_polygon_file)
    # crs(boreal_poly) <- crs(r[[1]])
    boreal_poly = prep_boreal_poly(boreal_polygon_file, r)
    
    df_band_full_list = list()
    df_band_boreal_list = list()
    df_band_north_list = list()
    
    for(bnum in bnum_list){
        bname = bnames[bnum]
        if(bname=='agbd_var'){
            # This creates a 'variance' band and gets the same stats, whcih include 'sum' needed later for full uncertainty accounting of a tile/boreal/north total variabne 
            add(r) = r[['agbd_sd']]^2   
            }
        
        ####
        # Full tile
        df_band_full = get_global_stats_df(r[[bnum]])

        ####
        # Cropped to boreal
        intersects <- terra::relate(r[[bnum]], boreal_poly, relation = "intersects")
        if(any(intersects)){
            #print('Tile intersects boreal')
            masked_r <- mask(r[[bnum]], boreal_poly)
            cropped_r <- crop(masked_r, boreal_poly)
            rm(masked_r)
            df_band_boreal = get_global_stats_df(cropped_r)
            #colnames(df_band_boreal) = paste0(bnames[bnum], '_boreal.', colnames(df_band_boreal))
            cnt_boreal = global(cropped_r, fun="notNA")$notNA
            rm(cropped_r)
            
        }else{
            #print('Tile outside of boreal')
            cnt_boreal = 0
            # Get empty df with the same names as the full df
            new_colnames = str_replace(colnames(df_band_full), 'full','boreal')
            #print(new_colnames)
            df_band_boreal <- data.frame(matrix(0, nrow = 1, ncol = length(colnames(df_band_full))))
            colnames(df_band_boreal) <- new_colnames
            #colnames(df_band_boreal) = colnames(df_band_full)
            #print(new_colnames)
        }

        #colnames(df_band_boreal) = paste0(bnames[bnum], '_boreal.', colnames(df_band_boreal))

        ####
        # Just above 51.6 - no cropping done if not needed
        results = crop_raster_to_north(r[[bnum]])
        cropped_r = results[[1]]
        DATA_WAS_CROPPED = results[[2]]
        
        if(DATA_WAS_CROPPED){
            #print('cropped')
            df_band_north = get_global_stats_df(cropped_r)
            cnt_north = global(cropped_r, fun="notNA")$notNA
        }else{
            #print('not cropped')
            cnt_north = 0  
            # Get empty df with the same names as the full df
            new_colnames = str_replace(colnames(df_band_full), 'full','north')
            df_band_north <- data.frame(matrix(0, nrow = 1, ncol = length(new_colnames)))
            colnames(df_band_north) <- new_colnames
            }
        
        # r_4326 <- project(r[[bnum]], "epsg:4326") #### TODO <-- does this change the area of a pixel?
        # # May be modified below depending on extent logic

        # #get mean and total for northern
        # if(ymin(r_4326)<51.6){  # should be =< ?
        #     if(ymax(r_4326)>51.6){
        #         #print('Tile straddles 51.6')
        #         new_ext <- ext(xmin(r_4326), xmax(r_4326), 51.6, ymax(r_4326))
        #         crop_r_4326 <- crop(r_4326, new_ext)
        #         df_band_north = get_global_stats_df(crop_r_4326)
        #         cnt_north = global(crop_r_4326, fun="notNA")$notNA
        #         rm(crop_r_4326)

        #     } else{
        #         #print('Tile completely south of 51.6')
        #         cnt_north = 0
        #         # northern_tile_total_agb <- 0.0
        #         # northern_tile_total_var <- 0.0
                
        #         # Get empty df with the same names as the full df
        #         new_colnames = str_replace(colnames(df_band_full), 'full','north')
        #         df_band_north <- data.frame(matrix(0, nrow = 1, ncol = length(new_colnames)))
        #         colnames(df_band_north) <- new_colnames
        #     }
        # } else {
        #     #print('Tile completely north of 51.6')
        #     cnt_north = global(r_4326, fun="notNA")$notNA
        #     # northern_tile_total_agb <- (0.09*global(mean_agb, 'sum', na.rm=TRUE)$sum)/1000000000
        #     # northern_tile_total_var <- global(var_agb, 'sum', na.rm=TRUE)$sum
        #     # Get df that matches the full one
        #     new_colnames = str_replace(colnames(df_band_full), 'full','north')
        #     df_band_north <- df_band_full
        #     colnames(df_band_north) <- new_colnames
        #     }

        df_band_full_list[[  paste0(bnames[bnum],'_full')  ]] = df_band_full
        df_band_boreal_list[[paste0(bnames[bnum],'_boreal')]] = df_band_boreal
        df_band_north_list[[paste0(bnames[bnum],'_north') ]] = df_band_north
        }
    
    # Get count of each extent (dont need for each band)
    cnt_full = global(r[[1]], fun="notNA")$notNA
    rm(r)
    
    # Concat final df
    df_tile = do.call('cbind', c(df_band_full_list, df_band_boreal_list, df_band_north_list))
    df_tile$tile_num = tile_num
    df_tile$cnt_full = cnt_full
    df_tile$cnt_boreal = cnt_boreal
    df_tile$cnt_north = cnt_north

    return(df_tile)
    
    }

compute_tile_stats <- function(tindex_fn, 
                               outdir_main='/projects/my-public-bucket/DPS_tile_lists/BOREAL_MAP',
                               start_idx=1, stop_idx=6000,
                               TILE_NUM=NA,
                               REDO=FALSE
                              ){
    #'''Return a csv for a list of tile rasters, run the tile-level dataframes of the raster band stats for each extent (full, boreal, north)
    #'''
    Sys.setenv("AWS_DEFAULT_REGION" = "us-west-2") 

    print(tindex_fn)

    Pg_to_Mg = 1e9
    ha_per_tile = 810000
    failed_tiles <- list()
    
    # Read the tindex from s3
    AGB_tindex_master <- aws.s3::s3read_using(read.csv, object = tindex_fn)

    if(!is.na(TILE_NUM)){
        print(paste0("Tile num: ", TILE_NUM))
        AGB_tindex_master = AGB_tindex_master %>% filter(tile_num == TILE_NUM)
        print(dim(AGB_tindex_master))
        }
    # AGB_tindex_master = AGB_tindex_master %>% filter(
    #                                                 (tile_num == 1630) | # no boreal no north
    #                                                 (tile_num == 1687) | # boreal no north
    #                                                  (tile_num == 1689) | # straddles 52.6 and boreal
    #                                                  (tile_num == 1188) | # boreal and north
    #                                                  (tile_num == 37743) # no boreal and north
    #                                                 ) ## test

    # Capture the version from the tindex path
    dps_version_str = str_split(tindex_fn,'/')[[1]][8] 
    map_version_str = str_split(tindex_fn,'/')[[1]][9]
    run_version_str = str_split(tindex_fn,'/')[[1]][10]
    
    # release_version_str should be our names v2.1, v2.1, v3.0 etc
    if(dps_version_str == 'dev_v1.5' && map_version_str == 'AGB_H30_2020'){release_version_str = 'v2.1'}
    if(dps_version_str == 'dev_v1.5' && map_version_str == 'AGB_S1H30_2020'){release_version_str = 'v2.2'}
    if(dps_version_str == 'v3.0.0'   && map_version_str == 'AGB_H30_2020'){release_version_str = 'v3.0'}

    print(paste(release_version_str, map_version_str, sep=' '))

    outdir = paste0(outdir_main, '/tile_stats_',map_version_str,'_',release_version_str)
    outdir_tiles = paste0(outdir,'/tiles')
    dir.create(outdir, showWarnings = FALSE)
    dir.create(outdir_tiles, showWarnings = FALSE)

    # for each tindex file, loop through to get the tile stats csv
    list_s3_path <- AGB_tindex_master$s3_path
    if(!is.na(TILE_NUM)){
        input_list = list_s3_path
        print(length(input_list))
        start_idx = 'tile_num' # just used in output csv name
        stop_idx = TILE_NUM # just used in output csv name
        total_list_len = 1
    }else{
        # Do this to help with multiprocessing
        input_list = list_s3_path[start_idx : stop_idx]
        total_list_len = length(input_list)
    }
    
    single_map_version_tile_stats_df_list = list()
    
    for(current_idx in seq_along(input_list) ){
        
        fn = input_list[current_idx]
        
        tryCatch({
            #print(fn)
            fn_base = unlist(str_split(basename(fn),'.tif'))[1]
            fn_parts_list = unlist(str_split(fn_base, '_'))
            tile_num = fn_parts_list[length(fn_parts_list)]

            out_tile_csv_fn = paste0(outdir_tiles,'/tile_stats_', sprintf("%07d", as.numeric(tile_num)),'.csv')
            
            # Check if file exists, skip if it does
            if (file.exists(out_tile_csv_fn)) {
                if(!REDO){
                    cat("Tile stats csv for : ",tile_num," already exists. Reading in from csv...\n")
                    single_map_version_tile_stats_df_list[[tile_num]] = read.csv(out_tile_csv_fn, row.names=NULL)
                    next  # Skip to next iteration
                }else{
                    cat("Re-doing csv for: ",tile_num,"\n")
                }
            }
    
            ###########################
            # Pixel-level summary stats (by tile) for band 1 (mean AGB) and band 2 (std AGB)
            # Extract Raster Data
            #raster_data = aws.s3::s3read_using(terra::rast, object = fn) # <-- this wont let values get read from raster_data
            #raster_data = terra::rast(str_replace(fn, "s3://","/vsis3/")) # <-- this will, but its slow
            #df = extract_raster_stats(terra::rast(str_replace(fn, "s3://","/vsis3/")), as.numeric(tile_num))
            df = do_tile_band_stats_df(terra::rast(str_replace(fn, "s3://","/vsis3/")), as.numeric(tile_num))
            
            ###########################
            # randomForest stats
            # These rF stats data are ONLY from the first rF model from all versions v3.0 and earlier
            #
            #df = setNames(data.frame(matrix(ncol = 3)),  c('tile_num', 'r2', 'rmse'))
            #print('Read RDS')
            
            stats = aws.s3::s3read_using(readRDS, object = gsub('.tif', '_stats.Rds' , fn))
            #print(fn)  
            #df$tile_num = as.numeric(tile_num)
            
            df$r2 = stats[[1]]
            df$rmse = stats[[2]]
            imp_sorted <- as.data.frame(stats$importance) %>% arrange(desc(IncNodePurity))
            imp_sorted$variable <- rownames(imp_sorted)
            df$imp_var_1 = imp_sorted$variable[1]
            df$imp_var_2 = imp_sorted$variable[2]
            df$imp_var_3 = imp_sorted$variable[3]
            df$imp_var_4 = imp_sorted$variable[4]
            df$imp_val_1 = imp_sorted$IncNodePurity[1]
            df$imp_val_2 = imp_sorted$IncNodePurity[2]
            df$imp_val_3 = imp_sorted$IncNodePurity[3]
            df$imp_val_4 = imp_sorted$IncNodePurity[4]
    
            ##############################
            # mean & sd of TILE TOTALS (from 250 iterations)
            #
            # TILE TOTAL AGB STOCK (MEAN & SD): totals summed across all pixels in [1] tile [2] boreal for each iterations (n=250)
            # These data are useful totals across any version
            #
            smry = aws.s3::s3read_using(read.csv, object = gsub('.tif', '_summary.csv' , fn))
            df$tile_total_agb_mean_pg   = mean(smry$tile_total)
            df$tile_total_agb_sd_pg     = sd(smry$tile_total) # This, divided by sqrt(n=250) is a standard error 
            df$boreal_total_agb_mean_pg = mean(smry$boreal_total)
            df$boreal_total_agb_sd_pg   = sd(smry$boreal_total) # This, divided by sqrt(n=250) is a standard error 
            ###############################
            # TILE TOTAL AGB DENSITY (MEAN & SD)
    
            df = df %>%
                        mutate(tile_total_agb_mean_mgha   = tile_total_agb_mean_pg *   (Pg_to_Mg / ha_per_tile),
                               tile_total_agb_sd_mgha     = tile_total_agb_sd_pg *     (Pg_to_Mg / ha_per_tile),
                               boreal_total_agb_mean_mgha = boreal_total_agb_mean_pg * (Pg_to_Mg / ha_per_tile),
                               boreal_total_agb_sd_mgha   = boreal_total_agb_sd_pg *   (Pg_to_Mg / ha_per_tile)
                              )

            # Write the tile level data frame for each tile
            write.csv(df, out_tile_csv_fn, row.names=FALSE)
            #write.csv(df, paste0(outdir,'/tile_stats_',map_version_str,'_',release_version_str,'_',current_idx,'_of_',total_list_len,EXPORT_TIME_STR,'.csv'))
            
            single_map_version_tile_stats_df_list[[tile_num]] = df
            
        }, error = function(e) {
            # Handle the error
            cat("Error with", fn, ":", e$message, "\n")
            
            # Save failed item to list
            failed_tiles <<- append(failed_tiles, list(list(
              fn = fn,
              error = e$message
            )))
          })
    }
    
    borealstats_df = do.call('rbind', single_map_version_tile_stats_df_list)
    
    borealstats_df$DPS_VERSION = dps_version_str
    borealstats_df$MAP_VERSION = map_version_str
    borealstats_df$RUN_VERSION = run_version_str
    borealstats_df$RELEASE_VERSION = release_version_str

    EXPORT_TIME_STR = format(Sys.time(), "_%Y%m%d")
    write.csv(borealstats_df, paste0(outdir,'/tile_stats_',map_version_str,'_',release_version_str,'_',start_idx,'_of_',stop_idx,EXPORT_TIME_STR,'.csv'), row.names=FALSE)
}

# # Get command line arguments
# args <- commandArgs(trailingOnly = TRUE)

# # Assign arguments to variables
# tindex_fn <- args[1]
# start_idx <- as.numeric(args[2])  # Convert to numeric if needed
# stop_idx <- as.numeric(args[3])  # Convert to numeric if needed

# compute_tile_stats(tindex_fn, start_idx=start_idx, stop_idx=stop_idx)