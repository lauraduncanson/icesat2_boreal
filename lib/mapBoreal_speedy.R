# This script was developed by Carlos A Silva and Laura Duncanson to produce tiled boreal biomass 30-m maps
# Inputs are stacks of Landsat and Copernicus DEM data, and tables of linked 30-m ATL08 data to coincident stack attrbitutes

#----------------------------------------------#

# 3.4 ICESat-2 biomass 30m ATL08

#### i) Description
#### Algorithm for tile-based AGBD mapping with ICESat-2, Copernicus DEM, and Landsat/HLS composites. This work was funded by NASA's ABoVE program, PI Duncanson, Co-Is Paul Montesano, Amy Neuenschwander, and NASA's ICESat-2 Science Team (PI Neuenschwader, Co-I Duncanson). Inputs and code from Nathan Thomas, Carlos Silva, Eric Guenther, Alex Mandel, and implementation assistance from NASA MAAP team George Change, Sujen Shah, Brian Satorius

#### ii) How the algorithm works?
#### Data tables of linked 30-m ATL08 RH metrics and covariate stack metrics are imported (outputs from tile_atl08.py)
#### AGBD models (externally developed) are loaded in R and applied over the ICESat-2 30m ATL08 data to data tables
#### A set of random forest models are fit to predict 30m ICESat-2 biomass as a function of covariates per tile
#### Raster stacks of covariates are sub-tiled to reduce memory usage
#### The suite of rf models are applied to each tile, mean and SD are output
#### Subtiles are recombined and written to disc as cogs

#### iii) Inputs
####  - rds_models: list of ICESat-2 simulation-derived AGB model paths
####  - models_id: models id
####  - stack: a combined raster stack of Landsat and Copernicus DEM data
####  - ice2_30_atl08: list containing the path to the data tables
####  - offset: offset applied in the model

#### iii) Outputs
####  COGs of predicted 30-m AGBD, SD AGBD

#----------------------------------------------#
############# functions ########################
#----------------------------------------------#
GEDI2AT08AGB<-function(rds_models,models_id, in_data, offset=100, DO_MASK=FALSE){
  
  # rds_models
  names(rds_models)<-models_id

  #if(DO_MASK){
  #    in_data = in_data %>% dplyr::filter(slopemask ==1 & ValidMask == 1)
  #}
    
  xtable_i<-na.omit(as.data.frame(in_data))
  names(xtable_i)[1:11]<- c("lon","lat","RH_25","RH_50","RH_60","RH_70","RH_75","RH_80","RH_90","RH_95","RH_98")

    #adjust for offset in model fits (100)
    #GEDI L4A team added offset to all the height metrics so they would never be negative)

  xtable_sqrt<-xtable_i[3:11]+offset

  # get unique ids
  # apply models by id
  xtable_sqrt$AGB<-NA
  xtable_sqrt$SE<-NA
    
  #Assign model id based on landcover

    # 1 = DBT trees (boreal-wide), 2=Evergreen needleleaf trees (boreal-wide), 12 = boreal-wide all PFT
    
    # for the seg_landcov: {0: "water", 1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest", \ 3: "deciduous needleleaf forest", 4: "deciduous broadleaf forest", \ 5: "mixed forest", 6: "closed shrublands", 7: "open shrublands", \ 8: "woody savannas", 9: "savannas", 10: "grasslands", 11: "permanent wetlands", \ 12: "croplands", 13: "urban-built", 14: "croplands-natural mosaic", \ 15: "permanent snow-ice", 16: "barren"})
    
    xtable_sqrt$model_id <- NA
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==0] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==1] <- 4
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==2] <- 4
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==3] <- 1
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==4] <- 1
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==5] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==6] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==7] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==8] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==9] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==10] <-12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==11] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==12] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==13] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==14] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==15] <- 12
    xtable_sqrt$model_id[xtable_sqrt$seg_landcov==16] <- 12
    
    xtable_sqrt$model_id<-names(rds_models)[1]
    ids<-unique(xtable_sqrt$model_id)

  for ( i in ids){
    
    # subset data for model id
    model_i<-readRDS(rds_models[names(rds_models)==i])
      
    # get variance covariance matrix
    model_varcov <- vcov(model_i)
    
    # get coefficients
    coeffs <- model_i$coefficients
    
    # modify coeffients through sampling variance covariance matrix
    mod.coeffs <- mvrnorm(n = 50, mu=coeffs, Sigma = model_varcov)

      print(mod.coeffs[1,])
    model_i$coefficients <- mod.coeffs[1,]

    # SE
    xtable_sqrt$SE[xtable_sqrt$model_id==i] <- summary(model_i)$sigma
    
    # AGB prediction
    xtable_sqrt$AGB[xtable_sqrt$model_id==i]<-predict(model_i, newdata=xtable_sqrt[xtable_sqrt$model_id==i,])
        
    #define C
    C <- mean(model_i$fitted.values)/mean(model_i$model$`sqrt(AGBD)`)
    
    #set negatives to zero
    negs <- which(xtable_sqrt$AGB<0)
    if(length(negs)>0){
        xtable_sqrt$AGB[negs] == 0.0
    }
      
    #we multiply by C in case there is a systematic over or under estimation in the model (bias correction)
    xtable_sqrt$AGB[xtable_sqrt$model_id==i]<-C*(xtable_sqrt$AGB[xtable_sqrt$model_id==i]^2)
      
    #set predictions where slopemask & validmask are 0 to 0
    xtable_sqrt$AGB[which(xtable_sqrt$slopemask==0)] <- 0.0
      
    xtable_sqrt$AGB[which(xtable_sqrt$ValidMask==0)] <- 0.0
      
    #set predictions where landcover is water, urban, snow, barren to 0
    bad_lc <- c(0, 13, 15, 16)
    xtable_sqrt$AGB[which(xtable_sqrt$seg_landcov %in% bad_lc)] <- 0.0

  }
    
  xtable2<-cbind(in_data, xtable_sqrt$AGB, xtable_sqrt$SE)
    ncol <- ncol(xtable2)
  colnames(xtable2)[(ncol-1):ncol]<-c('AGB', 'SE')
  return(xtable2)
}

# 2.7 ICESat-2 biomass algorithm development

#### i) Description
#### This section uses the sparse ICESat-2 AGB estimates and remote sensing covariates data (eg. extracted from ALOS-2, Landsat 8 OLI, Sentinel 2A...etc) for calibrating and testing AGB models using RF models

#### ii) How the algorithm works?
#### Users must to select the number of bootstrapping runs (ex. 100). In each run, the original dataset is divided into training (ex. 70%) and testing (ex. 30%) datastes for model calibration and validation. Users can select if they want to create the training and testing dataset using a random or stratified random sampling approach. R2, RMSE and MD are computed based on the training dataset. 

#### iii) Inputs
####  -rep: number of bootstrapping runs
####  -y: vector of the response variable in the dataset (ex. "g_agbm")
####  -x: dataframe containing the name of the covariates
####  -s_train: percentage used selected the training dataset (ex.70), 
####  -stack: stack of covar
####  -strat_random: If TRUE the original dataset will be splitted into training and testing using the stratified random approach (bins are defined by the quartiles). Otherwise, the random approach will be used. 

#### iii) Outputs
#### Maps of AGB (mean and sd)
#### Maps (mean and sd) of the prediction accuracy (RMSE, bias and R2 for the testig datasets)
#### Legend:
#### agb_mean = mean of the AGB predictions 
#### agb_sd = sd of the AGB predictions (n = rep)
#### armse_map_mean = mean of absolute RMSE values
#### armse_map_sd = sd of absolute RMSE values
#### rrmse_map_mean = mean of relative RMSE values
#### rrmse_map_sd = sd of relative RMSE values
#### abias_map_mean = mean of absolute RMSE values
#### abias_map_sd = sd of absolute RMSE values
#### rbias_map_mean = mean of relative of bias values
#### rbias_map_sd = sd of relative bias values
#### r2_map_mean = mean of r2 values
#### r2_map_sd = sd of r2 values 


#----------------------------------------------#
############# functions ########################
#----------------------------------------------#

# stats
StatModel <- function( obs, est){
  xy<-na.omit(cbind(obs,est))
  obs<-xy[,1]
  est<-xy[,2]
  rmse <- sqrt( sum(( est - obs )^2)/length(obs) ) # Root mean square error
  bias <- mean( est - obs ) # bias
  rmseR <- 100 * sqrt( sum(( est - obs )^2)/length(obs) ) / mean( obs )
  biasR <- 100 * mean( est - obs ) / mean( obs )
  r <- cor(est,obs)
  r2<-summary(lm(obs~est))$r.squared
  Stats<-data.frame( Stat=c("rmse","rmseR","bias","biasR","r","r2"),
                     Values=round(c(rmse,rmseR,bias,biasR,r,r2),2)) 
  return(Stats)
}

stratRandomSample<-function(agb=y,breaks, p){
  #require(data.table)
  n<-length(agb)
  ids<-1:n
  s<-round(n*p)
  agb[agb==0]<-0.0000000001
  ids_cut<-cut(agb,breaks=breaks, labels=F)
  df<-cbind(agb,ids,ids_cut)
  df<-data.table(df[!is.na(df[,1]),])
  number_sample<-ceiling(s/(length(breaks)-1))
  sel_all<-df[,.SD[sample(.N, min(number_sample,.N), replace = T)],by=ids_cut]
  return(ids_selected=sel_all$ids)
}

# modeling - fit a number of models and return as a list of models
agbModeling<-function(rds_models, models_id, in_data, pred_vars, offset=100, DO_MASK, se=NULL, rep=100,s_train=70, strat_random=TRUE,boreal_poly=boreal_poly, output){
    
    # apply GEDI models  
    xtable<-GEDI2AT08AGB(rds_models=rds_models,
                       models_id=models_id,
                       in_data=in_data, 
                       offset=offset,
                       DO_MASK=DO_MASK) 
    
    x <- xtable[pred_vars]
    y <- xtable$AGB
    se <- xtable$se
    
  stats_df<-NULL
  n<-nrow(x)
  ids<-1:n
  i.s=0
  model_list <- list()
    model_list <- list.append(model_list, xtable)
    
    # create one single rf using all the data; the first in model_list will be used for prediction
    fit.rf <- randomForest(y=y, x=x, ntree=500)
    model_list <- list.append(model_list, fit.rf)
    
  for (j in 2:rep){
    i.s<-i.s+1
      set.seed(j)
    if (strat_random==TRUE){
      trainRowNumbers<-stratRandomSample(agb=y,breaks=quantile(y, na.rm=T), p=s_train/100)
    } else {
      trainRowNumbers<-sort(sample(ids,round(n*s_train/100), T))
    }
    # Step 2: Create the training  dataset
    # select % of data to training and testing the models
    trainData.x <- x[trainRowNumbers,]
    trainData.y <- y[trainRowNumbers]
    
    # Step 3: Create the test dataset
    # select % of the data for validation
    testData.x <- x[!row.names(x) %in% trainRowNumbers,]
    testData.y <- y[!row.names(x) %in% trainRowNumbers]
    
    # rf modeling
    set.seed(j)
    
    fit.rf <- randomForest(y=trainData.y, x=trainData.x, ntree=500)
    pred.rf<-predict(fit.rf, newdata=testData.x) # 
    stats.rf<-cbind(method=rep("RF",6), rep=rep(j,6), StatModel(testData.y,pred.rf))
    
    # model validation stats
    stats_df<-rbind(stats_df,
                    stats.rf)
    row.names(stats_df)<-1:nrow(stats_df)
    
    #save output to a list where length = n(rep)
    model_list <- list.append(model_list, fit.rf)
    
  }
  return(model_list)
}

#split raster into subtiles, run mapping, recombine

# The function spatially aggregates the original raster
# it turns each aggregated cell into a polygon
# then the extent of each polygon is used to crop
# the original raster.
# The function returns a list with all the pieces
# in case you want to keep them in the memory. 
# it saves and plots each piece
# The arguments are:
# raster = raster to be chopped            (raster object)
# ppside = pieces per side                 (integer)

SplitRas <- function(raster,ppside,save){
  h        <- ceiling(ncol(raster)/ppside)
  v        <- ceiling(nrow(raster)/ppside)
  agg      <- aggregate(raster,fact=c(h,v))
  agg[]    <- 1:ncell(agg)
  agg_poly <- as.polygons(agg)
  names(agg_poly) <- "polis"
  r_list <- list()
  for(i in 1:ncell(agg)){
    e1          <- ext(agg_poly[agg_poly$polis==i,])
    r_list[[i]] <- crop(raster,e1)
  }
  if(save==T){
    for(i in 1:length(r_list)){
      writeRaster(r_list[[i]],filename=paste("SplitRas",i,sep=""),
                  format="GTiff",datatype="FLT4S",overwrite=TRUE)  
    }
  }
  return(r_list)
}


# mapping - apply the list of models to a set of sub-tiles, compute AGB, SD, 5th & 95th percentiles 

agbMapping<-function(x=x,y=y,model_list=model_list, tile_num=tile_num, stack=stack, boreal_poly=boreal_poly, output){
    #predict directly on raster using terra
    pred_stack <- na.omit(stack)
        
    map_pred <- predict(pred_stack, model_list[[1]], na.rm=TRUE)
    
    #set slope and valid mask to zero
    map_pred <- mask(map_pred, pred_stack$slopemask, maskvalues=0, updatevalue=0)
    
    map_pred <- mask(map_pred, pred_stack$ValidMask, maskvalues=0, updatevalue=0)

    #convert to total map (Pg, values per cell will be extremely small)
    total_convert <- function(x){(x*0.09)/1000000000}
    AGB_tot_map <- app(map_pred, total_convert)
    AGB_total <- global(AGB_tot_map, 'sum', na.rm=TRUE)$sum
    
    print('total AGB:')
    print(AGB_total)
    
    #calculate just the boreal total
    vect <- boreal_poly
    
    #for some reason can't figure this out with terra, reverting to raster
    library(raster)
    AGB_tot_map <- raster(AGB_tot_map)
    boreal_map <- mask(AGB_tot_map, vect, updatevalue=NA)
    detach("package:raster")  
    
    library(terra)
    boreal_map <- rast(boreal_map)
    AGB_tot_boreal <- global(boreal_map, 'sum', na.rm=TRUE)$sum
    rm(AGB_tot_map)
    rm(boreal_map)

    #loop over predicting for tile with each model in list
    for (i in 2:length(model_list)){
        fit.rf <- model_list[[i]]
        #create raster
        map_pred_temp <- predict(pred_stack, fit.rf, na.rm=TRUE)
        #set slope and valid mask to zero
        map_pred_temp <- mask(map_pred_temp, pred_stack$slopemask, maskvalues=0, updatevalue=0)
        map_pred_temp <- mask(map_pred_temp, pred_stack$ValidMask, maskvalues=0, updatevalue=0)
        map_pred_temp <- app(map_pred_temp, total_convert)
        AGB_total_temp <- global(map_pred_temp, 'sum', na.rm=TRUE)
        map_pred <- c(map_pred, map_pred_temp)
        AGB_total <- c(AGB_total, AGB_total_temp$sum)
        print('i:')
        print(i)
        
        #repeat for just boreal
        library(raster)
        map_pred_temp <- raster(map_pred_temp)
        boreal_map_temp <- mask(map_pred_temp, vect, updatevalue=NA)
        detach("package:raster") 
        rm(map_pred_temp)

        boreal_map_temp <- rast(boreal_map_temp)
        AGB_boreal_temp <- global(boreal_map_temp, 'sum', na.rm=TRUE)
        AGB_total_boreal <- c(AGB_tot_boreal, AGB_boreal_temp$sum)
        rm(boreal_map_temp)        
    }
    
    #take the average and sd per pixel
    mean_map <- app(map_pred, mean)
    sd_map <- app(map_pred, sd)
    
    print('individual')
    print(AGB_total)
    print(AGB_total_boreal)
    
    AGB_total_out <- as.data.frame(cbind(AGB_total, AGB_total_boreal))
    names(AGB_total_out) <- c('Tile_Total', 'Boreal_Total')


    out_fn_stem = paste("output/boreal_agb", format(Sys.time(),"%Y%m%d%s"), str_pad(tile_num, 4, pad = "0"), sep="_")
    out_fn_total <- paste0(out_fn_stem, '_total.csv')
    write.csv(file=out_fn_total, AGB_total_out)
    agb_maps <- c(mean_map, sd_map)
    
  return(agb_maps)
}


mapBoreal<-function(rds_models,
                    models_id,
                    ice2_30_atl08_path, 
                    ice2_30_sample_path,
                    offset=100,
                    s_train=70, 
                    rep=10,
                    ppside=2,
                    stack=stack,
                    strat_random=TRUE,
                    output,
                    minDOY=1,
                    maxDOY=365,
                    max_sol_el=0,
                    expand_training=TRUE,
                    local_train_perc=100,
                    min_n=3000,
                    DO_MASK=FALSE,
                    boreal_poly=boreal_poly){
    # Get tile num
    tile_num = tail(unlist(strsplit(path_ext_remove(ice2_30_atl08_path), "_")), n=1)
    print("Modelling and mapping boreal AGB")
    print(paste0("tile: ", tile_num))
    print(paste0("ATL08 input: ", ice2_30_atl08_path))
    
    #combine tables
    tile_data <- read.csv(ice2_30_atl08_path)
    
    #sub-sample tile data to n_tile
    n_tile <- as.double(min_n)
    
    #expand_training=TRUE when looking to expand to fulfill n_tile
    #expand_training=FALSE when looking to be explicity
    #default minDOY May 1 (121) maxDOY Sept 30 (273)
    default_minDOY <- 121
    default_maxDOY <- 273

    if(expand_training==TRUE){
        #first hard filtering
        filter1 <- which(tile_data$doy >= default_minDOY & tile_data$doy <= default_maxDOY & tile_data$sol_el < 0)
        n_filter1 <- length(filter1)

        #check if sufficient data, if not expand to max solar elevation allowed
        if(n_filter1 < n_tile){
            filter2 <- which(tile_data$doy >= minDOY & tile_data$doy <=maxDOY & tile_data$sol_el < max_sol_el)
            n_filter2 <- length(filter2)
            
            #check if n met, if not expand 1 month later in growing season, iteratively
            if(n_filter2 < n_tile){
                #check maxDOY
                temp_maxDOY <- default_maxDOY
                n_late <- 0
                for(late_months in 1:4){
                    if(n_late < n_tile){
                        temp_maxDOY <- default_maxDOY+(30*(late_months-1))
                        
                        if(temp_maxDOY < maxDOY){
                            filter_lateseason <- which(tile_data$doy > minDOY & tile_data$doy < temp_maxDOY & tile_data$sol_el < max_sol_el)
                            n_late <- length(filter_lateseason) 
                        }
                    }
                }
                print('n_late2:')
                print(n_late)

                if(n_late > n_tile){
                        tile_data <- tile_data[filter_lateseason,]
                } else{
                    #shift to iterative searching through early season
                    temp_minDOY <- default_minDOY
                    n_early <- 0
                    early_months <- 0
                    for(early_months in 1:4){
                        if(n_early < n_tile){
                            temp_minDOY <- default_minDOY-(30*(early_months-1))
                            
                            if(temp_minDOY > minDOY){
                                filter_earlyseason <- which(tile_data$doy >= temp_minDOY & tile_data$doy <=temp_maxDOY & tile_data$sol_el < max_sol_el)
                                n_early <- length(filter_earlyseason)
                            }
                        }

                    }
                    if(n_early > n_tile){
                        tile_data <- tile_data[filter_earlyseason,]
                    }
                         
                } 
            
            } else{
                tile_data <- tile_data[filter2,]
            }
        } else {
            tile_data <- tile_data[filter1,]
            }
    } else {
            #expand training = FALSE take defaults
            filter <- which(tile_data$doy >= default_minDOY & tile_data$doy <= default_maxDOY & tile_data$sol_el < 0)
            tile_data <- tile_data[filter,]
    }
        
    # Get rid of extra data
    n_avail <- nrow(tile_data)
    if(n_avail > n_tile){
        samp_ids <- seq(1,n_avail)
        tile_sample_ids <- sample(samp_ids, n_tile, replace=FALSE)
        tile_data <- tile_data[tile_sample_ids,]
    }
            
    #combine for fitting
    broad_data <- read.csv(ice2_30_sample_path)
    
    #take propertion of broad data we want based on local_train_perc
    sample_local <- n_tile * (local_train_perc/100)
    print(paste0('sample_local:', sample_local))
    sample_broad <- n_tile - sample_local
    print(paste0('sample_broad:', sample_broad))

    if(sample_local < n_tile){
        samp_ids <- seq(1,sample_local)
        tile_sample_ids <- sample(samp_ids, sample_local, replace=FALSE)
        tile_data <- tile_data[tile_sample_ids,]
    }
        
    if(sample_broad>0){
        samp_ids <- seq(1,sample_broad)
        broad_sample_ids <- sample(samp_ids, sample_broad, replace=FALSE)
        broad_data <- broad_data[broad_sample_ids,]
        all_train_data <- rbind(tile_data, broad_data)
    } else{
        all_train_data <- tile_data
    }
        
    print(paste0('table for model training generated with ', nrow(all_train_data), ' observations'))

    # run 
    if(DO_MASK){
        pred_vars <- c('slopemask', 'ValidMask', 'Xgeo', 'Ygeo','elevation', 'slope', 'tsri', 'tpi', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'SWIR2')
    }else{
        pred_vars <- c('Xgeo', 'Ygeo','elevation', 'slope', 'tsri', 'tpi', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'SWIR2')
    }
    
    #crs(agb.preds) <- crs(stack)
    models<-agbModeling(rds_models=rds_models,
                            models_id=models_id,
                            in_data=all_train_data,
                            pred_vars=pred_vars,
                            offset=offset,
                            DO_MASK=DO_MASK,
                            s_train=s_train,
                            rep=rep,
                            strat_random=strat_random,
                            boreal_poly=boreal_poly)
    
    xtable <- models[[1]]
    models <- models[-1]
    print(paste0('models successfully fit with ', length(pred_vars), ' predictor variables'))
        
    #split stack into list of iles
    if(ppside > 1){
        tile_list <- SplitRas(raster=stack,ppside=ppside,save=FALSE)
    
    print(paste0('tiles successfully split into ', length(tile_list), ' tiles'))

    #run mapping over each tile in a loop, create a list of tiled rasters for each layer

    n_subtiles <- length(tile_list)
    print(paste0('nsubtiles:', n_subtiles))
     for (tile in 1:n_subtiles){
        tile_stack <- tile_list[[tile]]

            maps<-agbMapping(x=xtable[pred_vars],
                     y=xtable$AGB,
                     model_list=models,
                     tile_num=tile_num,
                     stack=tile_stack,
                     boreal_poly=boreal_poly)
            
         if(tile == 1){out_map <- maps} 
         if(tile>1){
             out_map <- mosaic(maps, out_map, fun="max")
             rm(maps)
         }

        }
    
    }
    if (ppside == 1){
        out_map<-agbMapping(x=xtable[pred_vars],
                     y=xtable$AGB,
                     model_list=models,
                     tile_num=tile_num,
                     stack=stack,
                     boreal_poly=boreal_poly)
    }
    
    print('AGB successfully predicted!')
    
    print('mosaics completed!')

    # Make a 2-band stack as a COG

    out_fn_stem = paste("output/boreal_agb", format(Sys.time(),"%Y%m%d"), str_pad(tile_num, 4, pad = "0"), sep="_")

    #read in all csv files from output, combine for tile totals
    if(ppside > 1){
        #read csv files
        csv_files <- list.files(path='output/', pattern='*.csv', full.names=TRUE)
        n_files <- length(csv_files)
        for(h in 1:n_files){
            if(h==1){
                total_data <- read.csv(csv_files[h])
                file.remove(csv_files[h])

            }
            if(h>1){
                temp_data <- read.csv(csv_files[h])
                total_data <- rbind(total_data, temp_data)
                file.remove(csv_files[h])

            }    
        }
        
        #summarize accross subtiles
        total_AGB <- apply(total_data, 2, sum)
        #total_AGB_out <- cbind(total_AGB, total_AGB_boreal)
        #names(total_AGB_out) <- c('tile_total', 'tile_boreal_total')
        
        out_fn_stem = paste("output/boreal_agb", format(Sys.time(),"%Y%m%d%s"), str_pad(tile_num, 4, pad = "0"), sep="_")
        out_fn_total <- paste0(out_fn_stem, '_total.csv')
        write.csv(file=out_fn_total, total_AGB)
    }
    # Setup output filenames
    out_tif_fn <- paste(out_fn_stem, 'tmp.tif', sep="" )
    out_cog_fn <- paste(out_fn_stem, '.tif', sep="" )
    out_csv_fn <- paste0(out_fn_stem, '.csv' )
    out_stats_fn <- paste0(out_fn_stem, '_stats.csv', sep="")
    
    #set NA values
    #NAvalue(out_stack) <- 9999
    
    print(paste0("Write tmp tif: ", out_tif_fn))

    tifoptions <- c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6")
    writeRaster(out_map, filename=out_cog_fn, overwrite=TRUE, gdal=c("COMPRESS=NONE", "TFW=YES","of=COG"))
    
    #Change suggested from A Mandel to visualize cogs faster
    #rio cogeo create --overview-level=5 {input} {output} <- this is the system command that the below should be implementing
    #system2(command = "rio", 
    #    args    = c("cogeo", "create", "--overview-level=5", out_tif_fn, out_cog_fn), 
    #    stdout  = TRUE,
    #    stderr  = TRUE,
    #    wait    = TRUE
    #    )
    print(paste0("Write COG tif: ", out_cog_fn))
    
    #gdalUtils::gdal_translate(out_tif_fn, out_cog_fn, of = "COG")
    #file.remove(out_tif_fn)
          
     #Write out_table of ATL08 AGB as a csv
    out_table = xtable[,c('lon','lat','AGB','SE')]    
    write.csv(out_table, file=out_stats_fn)
    
    #write output for model accuracy and importance variables for single model
    #create one single model for stats
    rf_single <- randomForest(y=xtable$AGB, x=xtable[pred_vars], ntree=500, importance=TRUE)
    
    rsq <- max(rf_single$rsq, na.rm=T)
    rmse <- sqrt(min(rf_single$mse, na.rm=T))
    imp_vars <- rf_single$importance
    out_accuracy <- list(rsq, rmse, imp_vars)
    capture.output(out_accuracy, file=out_stats_fn)
    
    print("Returning names of COG and CSV...")
    return(list(out_cog_fn, out_csv_fn))
}

### Run code

# Get command line args
args = commandArgs(trailingOnly=TRUE)

#rds_filelist <- args[1]
data_table_file <- args[1]
topo_stack_file <- args[2]
l8_stack_file <- args[3]
DO_MASK_WITH_STACK_VARS <- args[4]
data_sample_file <- args[5]
iters <- args[6]
ppside <- args[7]
minDOY <- args[8]
maxDOY <- args[9]
max_sol_el <- args[10]
expand_training <- args[11]
local_train_perc <- args[12]
min_n <- args[13]
boreal_vect <- args[14]

ppside <- as.double(ppside)
minDOY <- as.double(minDOY)
maxDOY <- as.double(maxDOY)
max_sol_el <- as.double(max_sol_el)
local_train_perc <- as.double(local_train_perc)

MASK_LYR_NAMES = c('slopemask', 'ValidMask')
                          
MASK_LANDCOVER_NAMES = c(0,13,15,16)

print(paste0("Do mask? ", DO_MASK_WITH_STACK_VARS))

# loading packages and functions
#----------------------------------------------#
library(randomForest)
library(rgdal)
library(data.table)
library(ggplot2)
library(rlist)
library(fs)
library(stringr)
library(gdalUtils)
library(rockchalk)
library(terra)

# run code
# adding model ids
rds_models <- list.files(pattern='*.rds')

models_id<-names(rds_models)<-paste0("m",1:length(rds_models))

#use terra
topo <- rast(topo_stack_file)
l8 <- rast(l8_stack_file)

# make sure data are linked properly
stack<-crop(l8,ext(topo))

stack<-c(stack,topo)

if(DO_MASK_WITH_STACK_VARS){
    print("Masking stack...")
    # Bricking the stack will make the masking faster (i think)
    #brick = rast(stack)
    for(LYR_NAME in MASK_LYR_NAMES){
        m <- terra::subset(stack, grep(LYR_NAME, names(stack), value = T))
        
        stack <- mask(stack, m == 0, maskvalue=TRUE)

    }
    rm(m)
}

#read boreal polygon for masking later
print(boreal_vect)
boreal_poly <- readOGR(boreal_vect)

print("modelling begins")

maps<-mapBoreal(rds_models=rds_models,
                models_id=models_id,
                ice2_30_atl08_path=data_table_file, 
                ice2_30_sample=data_sample_file,
                offset=100.0,
                s_train=70, 
                rep=iters,
                ppside=ppside,
                stack=stack,
                strat_random=FALSE,
                output=out_fn,
                minDOY=minDOY,
                maxDOY=maxDOY,
                max_sol_el=max_sol_el,
                expand_training=expand_training,
                local_train_perc=local_train_perc,
                min_n=min_n,
                DO_MASK=DO_MASK_WITH_STACK_VARS,
                boreal_poly=boreal_poly)
