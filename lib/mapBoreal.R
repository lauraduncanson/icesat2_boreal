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
#### The suite of rf models are applied to each tile, mean, SD, p5 and p95 of predicted biomass are output
#### Subtiles are recombined and written to disc as cogs

#### iii) Inputs
####  - rds_models: list of ICESat-2 simulation-derived AGB model paths
####  - models_id: models id
####  - stack: a combined raster stack of Landsat and Copernicus DEM data
####  - ice2_30_atl08: list containing the path to the data tables
####  - offset: offset applied in the model

#### iii) Outputs
####  COGs of predicted 30-m AGBD, SD AGBD, 5th and 95th percentile estimates

#----------------------------------------------#
############# functions ########################
#----------------------------------------------#
GEDI2AT08AGB<-function(rds_models,models_id, in_data, offset=100, DO_MASK=FALSE){
  
  # rds_models
  names(rds_models)<-models_id
    
  # read table
  xtable<-in_data

  if(DO_MASK){
      xtable = xtable %>% dplyr::filter(slopemask ==1 & ValidMask == 1)
  }
    
  xtable_i<-na.omit(as.data.frame(xtable))
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
  }
    
  xtable2<-cbind(xtable, xtable_sqrt$AGB, xtable_sqrt$SE)
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
agbModeling<-function(x=x,y=y,se=NULL, rep=100,s_train=70, strat_random=TRUE,output){
  stats_df<-NULL
  n<-nrow(x)
  ids<-1:n
  i.s=0
  model_list <- list()
    
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
  agg_poly <- rasterToPolygons(agg)
  names(agg_poly) <- "polis"
  r_list <- list()
  for(i in 1:ncell(agg)){
    e1          <- extent(agg_poly[agg_poly$polis==i,])
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

agbMapping<-function(x=x,y=y,model_list=model_list, stack=stack,output){
    stack_df <- na.omit(as.data.frame(stack, xy=TRUE))
    stack_df$grid_id<-1:nrow(stack_df)
    stats_df<-NULL
    n<-nrow(x)
    ids<-1:n
    map_pred<-NULL

    #loop over predicting for tile with each model in list
    for (i in 2:length(model_list)){
        fit.rf <- model_list[[i]]
        map_i<-cbind(stack_df[,1:2],agb=predict(fit.rf, newdata=stack_df), rep=i, grid_id=stack_df$grid_id)
        map_pred<-rbind(map_pred,map_i)
    }
    
    #take the average and sd per pixel
    #set predictions to preds from the single rf model
    mean_map <- predict(model_list[[1]], newdata=stack_df)
    sd_map <-tapply(map_pred$agb,map_pred$grid_id,sd)
    p5 <- tapply(map_pred$agb, map_pred$grid_id, quantile, prob=0.05)
    p95 <- tapply(map_pred$agb, map_pred$grid_id, quantile, prob=0.95)
    
    #create output tile grid
    agb_maps <- rasterFromXYZ(cbind(stack_df[,1:2],
                                    agb_mean=mean_map,
                                    agb_sd=sd_map,
                                    p5=p5,
                                    p95=p95
                                  )
                              )
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
                    DO_MASK=FALSE){
    # Get tile num
    tile_num = tail(unlist(strsplit(path_ext_remove(ice2_30_atl08_path), "_")), n=1)
    print("Modelling and mapping boreal AGB")
    print(paste0("tile: ", tile_num))
    print(paste0("ATL08 input: ", ice2_30_atl08_path))
    
    #combine tables
    tile_data <- read.csv(ice2_30_atl08_path)
    
    #sub-sample tile data to n_tile
    n_tile <- 3000
    
    #optional filter for only night data
    #night_data <- which(tile_data$night_flg==1)
    #n_avail <- length(night_data)

    #if(n_avail > n_tile){
    #    tile_data <- tile_data[night_data,]
    #    samp_ids <- seq(1,n_avail)
    #    tile_sample_ids <- sample(samp_ids, n_tile, replace=FALSE)
    #    tile_data <- tile_data[tile_sample_ids,]
    #}
    
    #optional filter for solar elevation <5
    sol_e_data <- which(tile_data$sol_el < 5)
    n_avail <- length(sol_e_data)

    if(n_avail > n_tile){
        tile_data <- tile_data[sol_e_data,]
        samp_ids <- seq(1,n_avail)
        tile_sample_ids <- sample(samp_ids, n_tile, replace=FALSE)
        tile_data <- tile_data[tile_sample_ids,]
    }
    
    #combine for fitting
    broad_data <- read.csv(ice2_30_sample_path)
    
    #can combine with broad_data or not
    all_train_data <- tile_data
    #all_train_data <- rbind(tile_data, broad_data)
    
    #set boreal only models for specific weird tiles
    #weird_tiles <- c(4304, 4305, 4221, 4220, 1785, 1718, 1720, 1661, 1257, 1318, 1317, 1316, 1255, 1196, 949, 1062, 1063, 1005, 950, 1004)
    
    #if(tile_num %in% weird_tiles){
    #    all_train_data <- broad_data
    #}
    
    str(tile_data)
    str(broad_data)
    str(all_train_data)
    # apply GEDI models  
    xtable<-GEDI2AT08AGB(rds_models=rds_models,
                       models_id=models_id,
                       in_data=all_train_data, 
                       offset=offset,
                       DO_MASK=DO_MASK)
    #hist(xtable$AGB)
    print(paste0('table for model training generated with ', nrow(xtable), ' observations'))

    # run 
    if(DO_MASK){
        pred_vars <- c('Ygeo', 'Xgeo','elevation', 'slope', 'tsri', 'tpi', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW')
    }else{
        pred_vars <- c('Ygeo', 'Xgeo','elevation', 'slope', 'tsri', 'tpi', 'slopemask', 'Blue', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW')
    }
    
    #crs(agb.preds) <- crs(stack)
    models<-agbModeling(x=xtable[pred_vars],
                               y=xtable$AGB,
                               se=xtable$SE,
                               s_train=s_train,
                               rep=rep,
                               strat_random=strat_random)
    
    print(models[[1]])
    print(paste0('models successfully fit with ', length(pred_vars), ' predictor variables'))
    print(pred_vars)
        
    #split stack into list of iles
    tile_list <- SplitRas(raster=stack,ppside=ppside,save=FALSE)
    
    print(paste0('tiles successfully split into ', length(tile_list), ' tiles'))
    
    #run mapping over each tile in a loop, create a list of tiled rasters for each layer
    out_agb <- list()
    out_sd <- list()
    out_p5 <- list()
    out_p95 <- list()
    
     for (tile in 1:length(tile_list)){
        tile_stack <- tile_list[[tile]]
        #check if data in tile stack
        length_valid <- length(unique(tile_stack@data@values[,1]))
        if(length_valid > 2){
            print(tile)
            maps<-agbMapping(x=xtable[pred_vars],
                     y=xtable$AGB,
                     model_list=models,
                     stack=tile_stack)
        
            out_agb <- list.append(out_agb, as.list(maps)[[1]])
            out_sd <- list.append(out_sd, as.list(maps)[[2]])
            out_p5 <- list.append(out_p5, as.list(maps)[[3]])
            out_p95 <- list.append(out_p95, as.list(maps)[[4]])
            }
        }
    
    print('AGB successfully predicted!')
    #recombine tiles
    out_agb$fun   <- max
    agb.mosaic <- do.call(mosaic,out_agb)

    out_sd$fun   <- max
    sd.mosaic <- do.call(mosaic,out_sd)
    
    out_p5$fun   <- max
    p5.mosaic <- do.call(mosaic,out_p5)
    
    out_p95$fun   <- max
    p95.mosaic <- do.call(mosaic,out_p95)
    
    print('mosaics completed!')

    # Make a 4-band stack as a COG
    out_stack = stack(agb.mosaic, sd.mosaic, p5.mosaic, p95.mosaic)
    crs(out_stack) <- crs(tile_stack)
    out_fn_stem = paste("output/boreal_agb", format(Sys.time(),"%Y%m%d"), str_pad(tile_num, 4, pad = "0"), sep="_")
    #out_fn_stem = paste("/projects/testing/output/boreal_agb", format(Sys.time(),"%Y%m%d"), str_pad(tile_num, 4, pad = "0"), sep="_")

    # Setup output filenames
    out_tif_fn <- paste(out_fn_stem, 'tmp.tif', sep="_" )
    out_cog_fn <- paste(out_fn_stem, 'cog.tif', sep="_" )
    out_csv_fn <- paste0(out_fn_stem, '.csv' )
    out_stats_fn <- paste0(out_fn_stem, '_stats.csv', sep="_")
    
    print(paste0("Write tmp tif: ", out_tif_fn))

    tifoptions <- c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6")
    writeRaster(out_stack, filename=out_tif_fn, format="GTiff", datatype="FLT4S", overwrite=TRUE)
    
    #Change suggested from A Mandel to visualize cogs faster
    #rio cogeo create --overview-level=5 {input} {output} <- this is the system command that the below should be implementing
    #system2(command = "rio", 
    #    args    = c("cogeo", "create", "--overview-level=5", out_tif_fn, out_cog_fn), 
    #    stdout  = TRUE,
    #    stderr  = TRUE,
    #    wait    = TRUE
    #    )
    print(paste0("Write COG tif: ", out_cog_fn))
    
    gdalUtils::gdal_translate(out_tif_fn, out_cog_fn, of = "COG")
    file.remove(out_tif_fn)
          
    # LD's original return : a list of 2 things (both rasters)
    # Now, we can return a list of 3 things : the 2 maps, and the xtable (this has lat,lons, and AGB, SE for each ATL08 obs)
    #maps = append(agb.mosaic, list(xtable[,c('lon','lat','AGB','SE')]))
    
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

MASK_LYR_NAMES = c('slopemask', 'ValidMask')

print(paste0("Do mask? ", DO_MASK_WITH_STACK_VARS))

# loading packages and functions
#----------------------------------------------#
library(randomForest)
library(raster)
library(rgdal)
library(data.table)
library(ggplot2)
library(rlist)
library(fs)
library(stringr)
library(gdalUtils)

# run code
# adding model ids
rds_models <- list.files(pattern='*.rds')
print(rds_models)

models_id<-names(rds_models)<-paste0("m",1:length(rds_models))
print(topo_stack_file)

topo <- stack(topo_stack_file)
l8 <- stack(l8_stack_file)

# make sure data are linked properly
stack<-crop(l8,extent(topo)); stack<-stack(stack,topo)

if(DO_MASK_WITH_STACK_VARS){
    print("Making brick from stack and masking...")
    # Bricking the stack will make the masking faster (i think)
    brick = brick(stack)
    for(LYR_NAME in MASK_LYR_NAMES){
        m <- raster::subset(brick, grep(LYR_NAME, names(brick), value = T))
        brick <- mask(brick, m == 0, maskvalue=TRUE)
    }
    rm(m)
    rm(stack)
}

print("modelling begins")

maps<-mapBoreal(rds_models=rds_models,
                models_id=models_id,
                ice2_30_atl08_path=data_table_file, 
                ice2_30_sample=data_sample_file,
                offset=100.0,
                s_train=70, 
                rep=iters,
                ppside=ppside,
                stack=brick,
                strat_random=FALSE,
                output=out_fn,
                DO_MASK=DO_MASK_WITH_STACK_VARS)
