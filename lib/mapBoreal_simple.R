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

#applyModels takes a list of models and applies them to a raster stack, calculates per tile total (mean for height). 
applyModels <- function(models=models,
                           stack=stack,
                           pred_vars=pred_vars,
                           predict_var=predict_var,
                           tile_num){
        xtable <- models[[1]]
        models <- models[-1]
        rem <- length(models)
    
        if(rem>1){
            models <- models[-rem]
        }
    
        #create one single model for prediction
        if(predict_var=='AGB'){
            y <- xtable$AGB
        }
        if(predict_var=='Ht'){
            y <- xtable$RH_98
        }
        x <- xtable[pred_vars]
        print('fit general model')
        rf_single <- randomForest(y=y, x=x, ntree=500)
        #rf_single <- ranger(y=y, x=x, num.trees=500, oob.error=TRUE)
        pred_stack <- na.omit(stack)
        
        #subset just to layers of stack for prediction
        #pred_layer_names <- names(rf_single$forest$independent.variable.names)

        #pred_stack <- subset(pred_stack, pred_layer_names)
    
        print('apply first model to stack')

        agb_preds <- predict(pred_stack, models[[1]], na.rm=TRUE)
    
        print('mask first predictions')

        #set slope and valid mask to zero
        agb_preds <- mask(agb_preds, pred_stack$slopemask, maskvalues=0, updatevalue=0)
        print('mask first predictions 2')

        agb_preds <- mask(agb_preds, pred_stack$ValidMask, maskvalues=0, updatevalue=0)   
    
        print(paste0('models successfully applied with ', length(pred_vars), ' predictor variables'))
        
        #split stack into list of files
        if(ppside > 1){
            tile_list <- SplitRas(raster=stack,ppside=ppside,save=FALSE)
    
        print(paste0('tiles successfully split into ', length(tile_list), ' tiles'))

        #run mapping over each tile in a loop, create a list of tiled rasters for each layer

        n_subtiles <- length(tile_list)
        print(paste0('nsubtiles:', n_subtiles))
            if(exists('out_map')==TRUE){rm(out_map)}
            
         for (tile in 1:n_subtiles){
            tile_stack <- tile_list[[tile]]
            #for a subtile that is all NA, combine it with the next subtile
             print('tile number:')
             print(tile)
              if(predict_var=='AGB'){
                maps<-agbMapping(x=xtable[pred_vars],
                         y=y,
                         model_list=models,
                         tile_num=tile_num,
                         stack=tile_stack,
                         boreal_poly=boreal_poly)
                  
                if((exists('out_map')==FALSE) | tile==1){
                     if(length(maps)>1){
                         out_map <- maps[[1]]
                         tile_total <- maps[[2]]
                     }
                 } 

                 if((exists('out_map')==TRUE) & (length(maps)>1) & (tile>1)){
                     out_map <- mosaic(maps[[1]], out_map, fun="max")
                     if(exists('tile_total')==FALSE){tile_total <- 0.0}
                     print('tile total:')
                     tile_total <- tile_total + maps[[2]]
                     rm(maps)
                 }
             }
             
             if(predict_var=='Ht'){
                 maps<-HtMapping(x=xtable[pred_vars],
                         y=y,
                         model_list=models,
                         tile_num=tile_num,
                         stack=tile_stack,
                         boreal_poly=boreal_poly)
             
             if((exists('out_map')==FALSE) | tile==1){
                     if(length(maps)>1){
                         out_map <- maps[[1]]
                         tile_mean <- maps[[2]]$Tile_Mean
                         print(tile_mean)
                     }
                 } 

                 if((exists('out_map')==TRUE) & (length(maps)>1) & (tile>1)){
                     out_map <- mosaic(maps[[1]], out_map, fun="max")
                     if(exists('tile_mean')==FALSE){tile_mean <- 0.0}
                     print('tile mean:')
                     tile_mean <- mean(c(tile_mean, maps[[2]]$Tile_Mean))
                     print(tile_mean)
                     rm(maps)
                 }
             }   
            }
           }
        if (ppside == 1){
            if(predict_var=='AGB'){
               temp_map<-agbMapping(x=xtable[pred_vars],
                     y=y,
                     model_list=models,
                     tile_num=tile_num,
                     stack=stack,
                     boreal_poly=boreal_poly) 
            }
            if(predict_var=='Ht'){
                temp_map<-HtMapping(x=xtable[pred_vars],
                     y=y,
                     model_list=models,
                     tile_num=tile_num,
                     stack=stack,
                     boreal_poly=boreal_poly)
            }
            
            out_map <- temp_map[[1]]
            tile_total <- temp_map[[2]]
            rm(temp_map)
        }
        if(predict_var=='Ht'){
            out_data <- list(out_map, tile_mean)
        }
        if(predict_var=='AGB'){
            out_data <- list(out_map, tile_total)
        }
        return(out_data)
    }



combine_temp_files <- function(final_map, predict_var, tile_num){
        if(predict_var=='AGB'){
        tile_totals <- final_map[[2]]$Tile_Total
        
        #read in all csv files from output, combine for tile totals
            #read csv files
            csv_files <- list.files(path='output', pattern='_total.csv', full.names=TRUE)
            n_files <- length(csv_files)
            for(h in 1:n_files){
                if(h==1){
                    tile_data <- read.csv(csv_files[h])
                    total_data <- tile_data$Tile_Total
                    total_data_boreal <- tile_data$Boreal_Total
                    file.remove(csv_files[h])
                }
                if(h>1){
                    temp_data <- read.csv(csv_files[h])
                    total_data <- cbind(total_data, temp_data$Tile_Total)
                    total_data_boreal <- cbind(total_data_boreal, temp_data$Boreal_Total)
                    file.remove(csv_files[h])

                }    
            }
        
        
        #summarize accross models
        if(h>1){
            total_AGB <- apply(total_data, 1, sum, na.rm=TRUE)
            total_AGB_boreal <- apply(total_data_boreal, 1, sum, na.rm=TRUE) 
        } else {
            total_AGB <- sum(total_data, na.rm=TRUE)
            total_AGB_boreal <- sum(total_data_boreal, na.rm=TRUE)
        }
        
        total_AGB_out <- as.data.frame(cbind(total_AGB, total_AGB_boreal))
        names(total_AGB_out) <- c('tile_total', 'tile_boreal_total')
        
        out_fn_stem = paste("output/boreal_agb", format(Sys.time(),"%Y%m%d%s"), str_pad(tile_num, 4, pad = "0"), sep="_")
        out_fn_total <- paste0(out_fn_stem, '_total_all.csv')
        write.csv(file=out_fn_total, total_AGB_out, row.names=FALSE)
        combined_totals <- tile_totals

    }
    
    if(predict_var=='Ht'){
        print('Height successfully predicted!')
        print('Height mosaics completed!')
        tile_means <- final_map[[2]]

        # Make a 2-band stack as a COG

        out_fn_stem = paste("output/boreal_ht", format(Sys.time(),"%Y%m%d"), str_pad(tile_num, 4, pad = "0"), sep="_")
        
        #read in all csv files from output, combine for tile totals
            #read csv files
            csv_files <- list.files(path='output', pattern='_mean.csv', full.names=TRUE)
            n_files <- length(csv_files)
            for(h in 1:n_files){
                if(h==1){
                    tile_data <- read.csv(csv_files[h])
                    total_data <- tile_data$Tile_Mean
                    total_data_boreal <- tile_data$Boreal_Mean
                    file.remove(csv_files[h])
                }
                if(h>1){
                    temp_data <- read.csv(csv_files[h])
                    total_data <- cbind(total_data, temp_data$Tile_Mean)
                    total_data_boreal <- cbind(total_data_boreal, temp_data$Boreal_Mean)
                    file.remove(csv_files[h])
                }    
            }
            #summarize accross subtiles
            mean_Ht <- apply(total_data, 1, mean, na.rm=TRUE)
            mean_Ht_boreal <- apply(total_data_boreal, 1, mean, na.rm=TRUE)
            mean_Ht_out <- as.data.frame(cbind(mean_Ht, mean_Ht_boreal))
            names(mean_Ht_out) <- c('tile_mean', 'tile_boreal_mean')
        
            out_fn_stem = paste("output/boreal_ht", format(Sys.time(),"%Y%m%d%s"), str_pad(tile_num, 4, pad = "0"), sep="_")
            out_fn_total <- paste0(out_fn_stem, '_mean_all.csv')
            write.csv(file=out_fn_total, mean_Ht_out, row.names=FALSE)
            combined_totals <- tile_means
            print(str(tile_means))
    }
    return(combined_totals)
}

GEDI2AT08AGB<-function(rds_models,models_id, in_data, offset=100, DO_MASK=FALSE, one_model=TRUE){
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

    # if using ground photon models: 1 = DBT trees (boreal-wide), 4=Evergreen needleleaf trees (boreal-wide), 12 = boreal-wide all PFT
    # if using no ground photon models: 1 = DBT trees (boreal-wide), 3=Evergreen needleleaf trees (boreal-wide), 8 = boreal-wide all PFT

    # for the seg_landcov: {0: "water", 1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest", \ 3: "deciduous needleleaf forest", 4: "deciduous broadleaf forest", \ 5: "mixed forest", 6: "closed shrublands", 7: "open shrublands", \ 8: "woody savannas", 9: "savannas", 10: "grasslands", 11: "permanent wetlands", \ 12: "croplands", 13: "urban-built", 14: "croplands-natural mosaic", \ 15: "permanent snow-ice", 16: "barren"})
    
# for the seg_landcov update w v5 to Copernicus: {0: "NA", 111, 121: "evergreen needleleaf forest", 112, 122: "evergreen broadleaf forest", \ 113, 123: "deciduous needleleaf forest", 114, 124: "deciduous broadleaf forest", \ 115, 125: "mixed forest", 116, 126:"closed forest unknown", 20: "shrublands", 30: "herbaceous vegetation", \ 100: "moss and lichen", 60: "bare/sparse", 80, 200: "water", 40: "agriculture", 50: "urban-built", 70: "permanent snow-ice"})
    
    #xtable_sqrt$model_id <- NA
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==0] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==1] <- 3
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==2] <- 1
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==3] <- 3
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==4] <- 1
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==5] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==6] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==7] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==8] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==9] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==10] <-8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==11] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==12] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==13] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==14] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==15] <- 8
    #xtable_sqrt$model_id[xtable_sqrt$seg_landcov==16] <- 8
    
    #for no ground photon models
    xtable_sqrt$model_id <- 'NA'
    xtable_sqrt$model_id[xtable_i$seg_landcov==111] <- 'm3'
    xtable_sqrt$model_id[xtable_i$seg_landcov==121] <- 'm3'
    xtable_sqrt$model_id[xtable_i$seg_landcov==112] <- 'm1'
    xtable_sqrt$model_id[xtable_i$seg_landcov==122] <- 'm1'
    xtable_sqrt$model_id[xtable_i$seg_landcov==113] <- 'm3'
    xtable_sqrt$model_id[xtable_i$seg_landcov==123] <- 'm3'
    xtable_sqrt$model_id[xtable_i$seg_landcov==114] <- 'm1'
    xtable_sqrt$model_id[xtable_i$seg_landcov==124] <- 'm1'
    xtable_sqrt$model_id[xtable_sqrt$model_id=='NA'] <- 'm8'
    
    #for ground models, DBT coarse = m1, ENT coarse = m4, global = m12
    #xtable_sqrt$model_id <- 'NA'
    #xtable_sqrt$model_id[xtable_i$seg_landcov==111] <- 'm4'
    #xtable_sqrt$model_id[xtable_i$seg_landcov==121] <- 'm4'
    #xtable_sqrt$model_id[xtable_i$seg_landcov==112] <- 'm1'
    #xtable_sqrt$model_id[xtable_i$seg_landcov==122] <- 'm1'
    #xtable_sqrt$model_id[xtable_i$seg_landcov==113] <- 'm4'
    #xtable_sqrt$model_id[xtable_i$seg_landcov==123] <- 'm4'
    #xtable_sqrt$model_id[xtable_i$seg_landcov==114] <- 'm1'
    #xtable_sqrt$model_id[xtable_i$seg_landcov==124] <- 'm1'
    #xtable_sqrt$model_id[xtable_sqrt$model_id=='NA'] <- 'm12'

    #xtable_sqrt$model_id<-names(rds_models)[1]
    ids<-unique(xtable_sqrt$model_id)
    n_models <- length(ids)
    
    #one model for actual application - no resampling
    
    #iterate through re-sampling models
  for (i in ids){
    
    # subset data for model id
    model_i<-readRDS(rds_models[names(rds_models)==i])
      
    # get variance covariance matrix
    model_varcov <- vcov(model_i)
    
    # get coefficients
    coeffs <- model_i$coefficients
    
    # modify coeffients through sampling variance covariance matrix
    #if one_model = TRUE do not resample
    
      if(one_model==FALSE){
          mod.coeffs <- mvrnorm(n = 50, mu=coeffs, Sigma = model_varcov)
          model_i$coefficients <- mod.coeffs[1,]
      }

    # SE
    xtable_sqrt$SE[xtable_sqrt$model_id==i] <- summary(model_i)$sigma^2
    
    # AGB prediction
    xtable_sqrt$AGB[xtable_sqrt$model_id==i]<-predict(model_i, newdata=xtable_sqrt[xtable_sqrt$model_id==i,])
        
    #define C
    C <- mean(model_i$fitted.values^2)/mean(model_i$model$`sqrt(AGBD)`^2)
    
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
    #bad_lc <- c(0, 13, 15, 16)
    #update with copernicus
    bad_lc <- c(0, 60, 80, 200, 50, 70)
    xtable_sqrt$AGB[which(xtable_sqrt$seg_landcov %in% bad_lc)] <- 0.0

  }
    
  xtable2<-cbind(xtable_i, xtable_sqrt$AGB, xtable_sqrt$SE)
    ncol <- ncol(xtable2)
    colnames(xtable2)[(ncol-1):ncol]<-c('AGB', 'SE')
  return(xtable2)
}

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
agbModeling<-function(rds_models, models_id, in_data, pred_vars, offset=100, DO_MASK, se=NULL, rep=100,s_train=70, strat_random=TRUE,boreal_poly=boreal_poly, output, predict_var){
    # apply GEDI models for prediction
    xtable_predict<-GEDI2AT08AGB(rds_models=rds_models,
                       models_id=models_id,
                       in_data=in_data, 
                       offset=offset,
                       DO_MASK=DO_MASK, 
                       one_model=TRUE) 
    
    xtable<-GEDI2AT08AGB(rds_models=rds_models,
                       models_id=models_id,
                       in_data=in_data, 
                       offset=offset,
                       DO_MASK=DO_MASK, 
                       one_model=FALSE) 

  model_list <- list()
    model_list <- list.append(model_list, xtable_predict)
    if(predict_var=='AGB'){
        x <- xtable_predict[pred_vars]
        y <- xtable_predict$AGB
        se <- xtable$se

        fit.rf <- randomForest(y=y, x=x, ntree=1000, mtry=6)
        
        #print(max(fit.rf$rsq, na.rm=TRUE))
    }
    
    if(predict_var=='Ht'){
        x <- xtable_predict[pred_vars]
        y <- xtable_predict$RH_98
        se <- xtable$se
        # create one single rf using all the data; the first in model_list will be used for prediction
        print('fitting height model')
        
        #tune mtry
        #mtry_use <- tuneRF(x, y, ntreeTry=50, stepFactor=2, improve=0.05, trace=FALSE, plot=FALSE, doBest=FALSE)
        #fit the RF model that will actually be applied for mapping
        fit.rf <- randomForest(y=y, x=x, ntree=500)
        #print(max(fit.rf$rsq, na.rm=TRUE))
    }
    
    model_list <- list.append(model_list, fit.rf)
    
  stats_df<-NULL
  n<-nrow(x)
  ids<-1:n
  i.s=0

if(rep>1){    
    for (j in 1:rep){
    
    xtable<-GEDI2AT08AGB(rds_models=rds_models,
                       models_id=models_id,
                       in_data=in_data, 
                       offset=offset,
                       DO_MASK=DO_MASK, 
                       one_model=FALSE) 
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
    y_fit <- xtable$AGB[trainRowNumbers]
    x_fit <- xtable[pred_vars]
    x_fit <- x_fit[trainRowNumbers,]
    fit.rf <- randomForest(y=y_fit, x=x_fit, ntree=100)
    
    model_list <- list.append(model_list, fit.rf)  
      }
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
    rm(stack)
    
    if(length(unique(values(pred_stack$Red)))>1){
        map_pred <- predict(pred_stack, model_list[[1]], na.rm=TRUE)
        #set slope and valid mask to zero

        map_pred <- mask(map_pred, pred_stack$slopemask, maskvalues=0, updatevalue=0)
        map_pred <- mask(map_pred, pred_stack$ValidMask, maskvalues=0, updatevalue=0)   

        #convert to total map (Pg, values per cell will be extremely small)
        total_convert <- function(x){(x*0.09)/1000000000}
        AGB_tot_map <- app(map_pred, total_convert)
        AGB_total <- global(AGB_tot_map, 'sum', na.rm=TRUE)$sum
    
        #calculate just the boreal total
        boreal_total_temp <- extract(AGB_tot_map, boreal_poly, fun=sum, na.rm=TRUE)

        #AGB_total_boreal <- global(boreal_total_temp, 'sum', na.rm=TRUE)$sum

        AGB_total_boreal <- sum(boreal_total_temp$lyr.1, na.rm=TRUE)
        print(AGB_total_boreal)
        rm(AGB_tot_map)
        n_models <- length(model_list)
        if(n_models>1){
            #loop over predicting for tile with each model in list
        for (i in 2:length(model_list)){
            fit.rf <- model_list[[i]]
            
            #create raster
            map_pred_temp <- predict(pred_stack, fit.rf, na.rm=TRUE)
            
            #set slope and valid mask to zero

            map_pred_temp <- mask(map_pred_temp, pred_stack$slopemask, maskvalues=0, updatevalue=0)
            map_pred_temp <- mask(map_pred_temp, pred_stack$ValidMask, maskvalues=0, updatevalue=0)

            map_pred_tot_temp <- app(map_pred_temp, total_convert)
            AGB_total_temp <- global(map_pred_tot_temp, 'sum', na.rm=TRUE)$sum
            
            map_pred <- c(map_pred, map_pred_temp)
            AGB_total <- c(AGB_total, AGB_total_temp)
        
            #repeat for just boreal
            #boreal_map_temp <- mask(map_pred_tot_temp, boreal_poly, updatevalue=0)

            boreal_total_temp <- extract(map_pred_tot_temp, boreal_poly, fun=sum, na.rm=TRUE)

            rm(map_pred_tot_temp)
            rm(map_pred_temp)

            #AGB_boreal_temp <- global(boreal_map_temp$lyr1, 'sum', na.rm=TRUE)$sum
            AGB_boreal_temp <- sum(boreal_total_temp$lyr.1, na.rm=TRUE)
            print(AGB_boreal_temp)
            AGB_total_boreal <- c(AGB_total_boreal, AGB_boreal_temp)
        }
    #take the average and sd per pixel
    #mean_map <- app(map_pred, mean)
    sd_map <- app(map_pred, sd)
    }
    
    #model with all data for mapping
    mean_map <- map_pred[[1]]
    
  #else {
   #     #this is if the entire sub-tile is NA
   #     AGB_total <- rep(0,length(model_list))
   #     AGB_total_boreal <- rep(0,length(model_list))
   #     mean_map <- pred_stack$slopemask
   #     sd_map <- pred_stack$slopemask
    #}    
    AGB_total_out <- as.data.frame(cbind(AGB_total, AGB_total_boreal))
    names(AGB_total_out) <- c('Tile_Total', 'Boreal_Total')
    out_fn_stem = paste("output/boreal_agb", format(Sys.time(),"%Y%m%d%s"), str_pad(tile_num, 4, pad = "0"), sep="_")

    out_fn_total <- paste0(out_fn_stem, '_total.csv')

    write.csv(file=out_fn_total, AGB_total_out, row.names=FALSE)
    if(n_models>1){
        agb_maps <- list(c(mean_map, sd_map, map_pred), AGB_total_out)
    } else{
        agb_maps <- list(c(mean_map), AGB_total_out)
    }

  return(agb_maps)
  }
}


HtMapping<-function(x=x,y=y,model_list=model_list, tile_num=tile_num, stack=stack, boreal_poly=boreal_poly, output){
    #predict directly on raster using terra
    pred_stack <- na.omit(stack)

    if(length(unique(values(pred_stack$Red)))>1){
        map_pred <- predict(pred_stack, model_list[[1]], na.rm=TRUE)
        #set slope and valid mask to zero
        map_pred <- mask(map_pred, pred_stack$slopemask, maskvalues=0, updatevalue=0)
        map_pred <- mask(map_pred, pred_stack$ValidMask, maskvalues=0, updatevalue=0)   

        Ht_mean <- global(map_pred, 'mean', na.rm=TRUE)$mean
        print(Ht_mean)
    
        #calculate just the boreal total
        boreal_ht_temp <- extract(map_pred, boreal_poly, na.rm=TRUE)
        Ht_mean_boreal <- mean(boreal_ht_temp$lyr1, na.rm=TRUE)
        print(Ht_mean_boreal)
        rm(boreal_ht_temp)
        
        mean_map <- map_pred[[1]]

    #loop over predicting for tile with each model in list
    n_models <- length(model_list)
    print('n_models:')
    print(n_models)
    if(n_models>1){
        for (i in 2:length(model_list)){
        fit.rf <- model_list[[i]]
        #create raster
        map_pred_temp <- predict(pred_stack, fit.rf, na.rm=TRUE)
        #set slope and valid mask to zero
        map_pred_temp <- mask(map_pred_temp, pred_stack$slopemask, maskvalues=0, updatevalue=0)
        map_pred_temp <- mask(map_pred_temp, pred_stack$ValidMask, maskvalues=0, updatevalue=0)
        
        Ht_mean_temp <- global(map_pred_temp, 'mean', na.rm=TRUE)$mean
        map_pred <- c(map_pred, map_pred_temp)
        Ht_mean <- c(Ht_mean, Ht_mean_temp)
        
        #repeat for just boreal
        boreal_ht_temp <- extract(map_pred_temp, boreal_poly, na.rm=TRUE)
        rm(map_pred_temp)

        Ht_boreal_temp <- mean(boreal_ht_temp$lyr1, na.rm=TRUE)
        print(Ht_boreal_temp)
        Ht_mean_boreal <- c(Ht_mean_boreal, Ht_boreal_temp)
        rm(boreal_ht_temp)        
        }
    #take the average and sd per pixel
    mean_map <- app(map_pred, mean)
    sd_map <- app(map_pred, sd)
    }
    
    #model with all data for mapping
 
    Ht_mean_out <- as.data.frame(cbind(Ht_mean, Ht_mean_boreal))
    names(Ht_mean_out) <- c('Tile_Mean', 'Boreal_Mean')

    out_fn_stem = paste("output/boreal_ht", format(Sys.time(),"%Y%m%d%s"), str_pad(tile_num, 4, pad = "0"), sep="_")

    out_fn_total <- paste0(out_fn_stem, '_mean.csv')

    write.csv(file=out_fn_total, Ht_mean_out, row.names=FALSE)
    if(n_models>1){
        ht_maps <- list(c(mean_map, sd_map), Ht_mean_out)
    } else{
        ht_maps <- list(c(mean_map), Ht_mean_out)
    }

  return(ht_maps)
  }
}

check_var <- function(totals){
        #calc sd iteratively
        sd <- totals*0.0
        nrow <- length(totals)
        print('nrow:')
        print(nrow)
        n <- seq(1, nrow)
        for (i in 1:nrow){
            temp_tot <- totals[1:i]
            if(i>2){
                 sd[i] <- sd(temp_tot, na.rm=TRUE)
            }
        }
        #test the variance for the last 10 sds
            #b_min <- (nrow-20)
            b_max <- (nrow-9)
            baseline_var <- mean(sd[1:b_max], na.rm=TRUE)
            print('baseline_var:')
            print(baseline_var)
            
            last_var <- mean(sd[1:nrow], na.rm=TRUE)
            print(last_var)
            var_diff <- abs((baseline_var - last_var)/baseline_var)
            print('var_diff:')
            print(var_diff)
            return(var_diff)
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
                    boreal_poly=boreal_poly,
                    predict_var,
                    max_n=3000){

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
        
    # Get rid of extra data above max_n
    n_avail <- nrow(tile_data)
    
    if(n_avail > max_n){
        samp_ids <- seq(1,n_avail)
        tile_sample_ids <- sample(samp_ids, n_tile, replace=FALSE)
        tile_data <- tile_data[tile_sample_ids,]
    }
            
    #combine for fitting
    broad_data <- read.csv(ice2_30_sample_path)
    
    
    #remove first col of broad_data
    #broad_data <- broad_data[,2:ncol(broad_data)]
    #take propertion of broad data we want based on local_train_perc
    sample_local <- n_tile * (local_train_perc/100)
    
    #if static broad, use all local train data
    #sample_local <- n_tile
    print(n_tile)
    print('sample_local:')
    print(n_avail)
    if(sample_local < n_tile){
        samp_ids <- seq(1,sample_local)
        tile_sample_ids <- sample(samp_ids, sample_local, replace=FALSE)
        tile_data <- tile_data[tile_sample_ids,]
    }

    #sample from broad data to complete sample size
    #this will work if either there aren't enough local samples for n_min OR if there is forced broad sampling
    n_broad <- n_tile - nrow(tile_data)
    if(n_broad > 1){
        broad_samp_ids <- seq(1,n_broad)
        
        #subset broad data to be within a certain latitude
        lat_thresh <- 5
        min_lat <- min(tile_data$lat)
        broad_in_lat <- which(broad_data$lat > (min_lat-lat_thresh) & broad_data$lat < (min_lat+lat_thresh))
        broad_data <- broad_data[broad_in_lat,]
        broad_sample_ids <- sample(broad_samp_ids, n_broad, replace=FALSE)
        broad_data <- broad_data[broad_sample_ids,]
        str(tile_data)
        str(broad_data)
        all_train_data <- rbind(tile_data, broad_data)
    } else {
        all_train_data <- tile_data
        }
    
    #remove first col
    all_train_data <- all_train_data[,-1]
    
    tile_data_output <- tile_data[,-1]
        
    print(paste0('table for model training generated with ', nrow(all_train_data), ' observations'))

    # run 
    if(DO_MASK){
        pred_vars <- c('slopemask', 'ValidMask', 'Red', 'Green','elevation', 'slope', 'tsri', 'tpi', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'SWIR2')

    }else{
        pred_vars <- c('Xgeo', 'Ygeo','elevation', 'slope', 'tsri', 'tpi', 'Green', 'Red', 'NIR', 'SWIR', 'NDVI', 'SAVI', 'MSAVI', 'NDMI', 'EVI', 'NBR', 'NBR2', 'TCB', 'TCG', 'TCW', 'SWIR2')
    }
print(predict_var)
    models<-agbModeling(rds_models=rds_models,
                            models_id=models_id,
                            in_data=all_train_data,
                            pred_vars=pred_vars,
                            offset=offset,
                            DO_MASK=DO_MASK,
                            s_train=s_train,
                            rep=rep,
                            strat_random=strat_random,
                            boreal_poly=boreal_poly,
                            predict_var=predict_var)
    
    print('model fitting complete!')

    final_map <- applyModels(models, stack, pred_vars, predict_var, tile_num)
    
    xtable <- models[[1]]

    if(ppside > 1){
        combined_totals <- combine_temp_files(final_map, predict_var, tile_num)
    }
    
    #subset out the iteration bands
    out_map_all <- subset(final_map[[1]], 3:nlyr(final_map[[1]]))
    
    #just pull the mean for out_map, sd will be added later
    out_map <- subset(final_map[[1]], 1)
    rm(final_map)

    #set the variance threshold - 0.05 = 5%
    var_thresh <- 0.05
    
    if(rep>1){
        var_diff <- check_var(combined_totals)
        print('var_diff:')
        print(var_diff)

        #if larger difference, need more models and more iterations
        #save(combined_totals, file='/projects/lduncanson/testing/test_totals.Rdata')
        #set some maximum number of iterations
        max_iters <- 200
        if(length(combined_totals)<max_iters){
            while(var_diff > var_thresh){
            print('Adding more interations...')
            new_models <- agbModeling(rds_models=rds_models,
                            models_id=models_id,
                            in_data=all_train_data,
                            pred_vars=pred_vars,
                            offset=offset,
                            DO_MASK=DO_MASK,
                            s_train=s_train,
                            rep=10,
                            strat_random=strat_random,
                            boreal_poly=boreal_poly,
                            predict_var=predict_var)
                
            new_final_map <- applyModels(new_models, stack, pred_vars, predict_var, tile_num)
            
            if(ppside > 1){
                combined_totals_new <- combine_temp_files(new_final_map, predict_var, tile_num)
            }
                
            temp <- new_final_map[[2]]
            
            #combine original map with new iterations map
            out_map_all <- c(out_map_all, subset(new_final_map[[1]], 3:nlyr(new_final_map[[1]])))
                
                if(predict_var=='AGB'){
                    new_tile_totals <- new_final_map[[2]]$Tile_Total
                }
                if(predict_var=='Ht'){
                    new_tile_totals <- new_final_map[[2]]$Tile_Mean
                }    
            rm(new_final_map)
            combined_totals <- c(combined_totals, combined_totals_new)
            var_diff <- check_var(combined_totals)
                print('check length')
                print(length(combined_totals))
            if(length(combined_totals)>75){
                var_thresh <- 0.06
                }
            if(length(combined_totals)>100){
                var_thresh <- 0.08
                }
            if(length(combined_totals)>200){
                var_thresh <- 0.1
                }
            }
        }
    }                             
    
    #combine all output total files into one
    if(predict_var=='AGB'){
        #read csv files
        csv_files <- list.files(path='output', pattern='_total_all.csv', full.names=TRUE)
        n_files <- length(csv_files)
        print('length all files:')
        print(n_files)
        for(h in 1:n_files){
            if(h==1){
                tile_data <- read.csv(csv_files[h])
                file.remove(csv_files[h])
            }
            if(h>1){
                temp_data <- read.csv(csv_files[h])
                tile_data <- rbind(tile_data, temp_data)
                file.remove(csv_files[h])
            }    
        }
       
        total_AGB_out <- as.data.frame(tile_data)
        names(total_AGB_out) <- c('tile_total', 'tile_boreal_total')
        
        out_fn_stem = paste("output/boreal_agb", format(Sys.time(),"%Y%m%d%s"), str_pad(tile_num, 4, pad = "0"), sep="_")
        out_fn_total <- paste0(out_fn_stem, '_total_iters.csv')
        write.csv(file=out_fn_total, total_AGB_out, row.names=FALSE)
    
    }
    if(predict_var=='Ht'){
         #read csv files
        csv_files <- list.files(path='output', pattern='_mean_all.csv', full.names=TRUE)
        n_files <- length(csv_files)
        print('length all files:')
        print(n_files)
        for(h in 1:n_files){
            if(h==1){
                tile_data <- read.csv(csv_files[h])
                file.remove(csv_files[h])
            }
            if(h>1){
                temp_data <- read.csv(csv_files[h])
                tile_data <- rbind(tile_data, temp_data)
                file.remove(csv_files[h])
            }    
        }
       
        mean_AGB_out <- as.data.frame(tile_data)
        names(mean_AGB_out) <- c('tile_mean', 'tile_boreal_mean')
        
        out_fn_stem = paste("output/boreal_ht", format(Sys.time(),"%Y%m%d%s"), str_pad(tile_num, 4, pad = "0"), sep="_")
        out_fn_total <- paste0(out_fn_stem, '_mean_iters.csv')
        write.csv(file=out_fn_total, mean_AGB_out, row.names=FALSE)
        
    }
        
        print('AGB successfully predicted!')
    
        print('mosaics completed!')
    
    # Setup output filenames
    out_tif_fn <- paste(out_fn_stem, 'tmp.tif', sep="" )
    out_cog_fn <- paste(out_fn_stem, '.tif', sep="" )
    out_csv_fn <- paste0(out_fn_stem, '.csv' )
    out_train_fn <- paste0(out_fn_stem, '_train_data.csv', sep="")
    out_stats_fn <- paste0(out_fn_stem, '_stats.Rds', sep="")
    out_model_fn <- paste0(out_fn_stem, '_model.Rds', sep="")

    print(paste0("Write tmp tif: ", out_tif_fn))
    #change -9999 to NA
    #out_map <- classify(out_map, cbind(-9999.000, NA))
    out_map <- subst(out_map, -9999, NA)
    out_sd <- app(out_map_all, sd)
    out_sd <- subst(out_sd, -9999, NA)
    out_map <- c(out_map, out_sd)
                  
    NAflag(out_map)
    
    tifoptions <- c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6", "OVERVIEW_RESAMPLING=AVERAGE")
    writeRaster(out_map, filename=out_cog_fn, filetype="COG", gdal=c("COMPRESS=LZW", overwrite=TRUE, gdal=c("COMPRESS=LZW", "OVERVIEW_RESAMPLING=AVERAGE")))
    #writeRaster(out_map, filename=out_cog_fn, overwrite=TRUE)
    
    print(paste0("Write COG tif: ", out_cog_fn))
          
    nrow_tile <- nrow(tile_data_output)

     #Write out_table of ATL08 AGB as a csv
    if(predict_var=='AGB'){

        out_table <- xtable[,c('lon', 'lat', 'AGB', 'SE')]
        write.csv(out_table, file=out_train_fn, row.names=FALSE)
        rf_single <- randomForest(y=xtable$AGB, x=xtable[pred_vars], ntree=1000, importance=TRUE, mtry=6)
        local_model <- lm(rf_single$predicted[1:nrow_tile] ~ xtable$AGB[1:nrow_tile], na.rm=TRUE)

    }
    
    if(predict_var=='Ht'){
        out_table = xtable[c('lon','lat','RH_98')]    
        write.csv(out_table, file=out_train_fn, row.names=FALSE)
        rf_single <- randomForest(y=xtable$RH_98, x=xtable[pred_vars], ntree=1000, importance=TRUE, mtry=6)
        local_model <- lm(rf_single$predicted[1:nrow_tile] ~ xtable$RH_98[1:nrow_tile], na.rm=TRUE)

    }

    #write output for model accuracy and importance variables for single model
    #create one single model for stats
    
    saveRDS(rf_single, file=out_model_fn)
    rsq <- max(rf_single$rsq, na.rm=T)
    print('rsq:')
    print(rsq)
    #calc rsq only over local data
    rsq_local <- summary(local_model)$r.squared
    print('r
max_iters <- sq_local:')
    print(rsq_local)
    
    na_data <- which(is.na(local_model$predicted==TRUE))

    if(length(na_data)>1){
        #rmse_local <- sqrt(mean(local_model$residuals[-na_data]^2))
    } else {
        rmse_local <- sqrt(mean(local_model$residuals^2))
    }
    print('rmse_local:')
    print(rmse_local)
    
    imp_vars <- rf_single$importance
    out_accuracy <- list(rsq_local, rmse_local, imp_vars)
    saveRDS(out_accuracy, file=out_stats_fn)
    
    print("Returning names of COG and CSV...")
    return(list(out_cog_fn, out_csv_fn))
}

####################### Run code ##############################

# Get command line args
args = commandArgs(trailingOnly=TRUE)

#rds_filelist <- args[1]
data_table_file <- args[1]
topo_stack_file <- args[2]
l8_stack_file <- args[3]
LC_mask_file <- args[4]
DO_MASK_WITH_STACK_VARS <- args[5]
data_sample_file <- args[6]
iters <- args[7]
ppside <- args[8]
minDOY <- args[9]
maxDOY <- args[10]
max_sol_el <- args[11]
expand_training <- args[12]
local_train_perc <- args[13]
min_n <- args[14]
boreal_vect <- args[15]
predict_var <- args[16]
max_n <- args[17]

#for debugging replace args with hard paths
#data_table_file <- '/projects/my-private-bucket/dps_output/run_tile_atl08_ubuntu/tile_atl08/2022/11/30/19/22/04/120959/atl08_005_30m_filt_topo_landsat_20221130_1216.csv'
#topo_stack_file <- '/projects/shared-buckets/nathanmthomas/alg_34_testing/Copernicus_1216_covars_cog_topo_stack.tif'
#l8_stack_file <- '/projects/shared-buckets/nathanmthomas/alg_34_testing/HLS_1216_06-15_09-01_2019_2021.tif'
#LC_mask_file <- '/projects/shared-buckets/nathanmthomas/alg_34_testing/esa_worldcover_v100_2020_1216_cog.tif'
#DO_MASK_WITH_STACK_VARS <- 'TRUE'
#data_sample_file <- '/projects/my-private-bucket/boreal_train_data_v11.csv'
#iters <- 1
#ppside <- 2
#minDOY <- 130
#maxDOY <- 250
#max_sol_el <- 5
#expand_training <- 'TRUE'
#local_train_perc <- 100
#min_n <- 5000
#boreal_vect <- '/projects/shared-buckets/nathanmthomas/boreal_tiles_v003.gpkg'
#predict_var <- 'AGB'

ppside <- as.double(ppside)
minDOY <- as.double(minDOY)
maxDOY <- as.double(maxDOY)
max_sol_el <- as.double(max_sol_el)
local_train_perc <- as.double(local_train_perc)

MASK_LYR_NAMES = c('slopemask', 'ValidMask')
                          
#MASK_LANDCOVER_NAMES = c(0,13,15,16)
MASK_LANDCOVER_NAMES = c(50,70,80,100)

print(paste0("Do mask? ", DO_MASK_WITH_STACK_VARS))

# loading packages and functions
#----------------------------------------------#
library(randomForest)
#library(rgdal)
library(data.table)
library(ggplot2)
library(rlist)
library(fs)
library(stringr)
#library(gdalUtils)
library(rockchalk)
library(terra)
# run code
# adding model ids
rds_models <- list.files(pattern='*.rds')

models_id<-names(rds_models)<-paste0("m",1:length(rds_models))

#use terra
topo <- rast(topo_stack_file)
l8 <- rast(l8_stack_file)
lc <- rast(LC_mask_file)

# make sure data are linked properly
#check extents
nrow_topo = nrow(topo)
nrow_l8 = nrow(l8)
nrow_diff <- abs(nrow_topo-nrow_l8)
print(nrow_diff)

ncol_topo <- ncol(topo)
ncol_l8 <- ncol(l8)
ncol_diff <- abs(ncol_topo-ncol_l8)

if(nrow_diff>0 || ncol_diff>0){
   #resample
    topo <- resample(topo, l8, method='near')
    lc <- resample(lc, l8, method='near')
} 

ext(topo) <- ext(l8)
ext(lc) <- ext(l8)

stack<-c(l8,topo, lc)

if(DO_MASK_WITH_STACK_VARS){
    print("Masking stack...")
    # Bricking the stack will make the masking faster (i think)
    #brick = rast(stack)
    for(LYR_NAME in MASK_LYR_NAMES){
        m <- terra::subset(stack, grep(LYR_NAME, names(stack), value = T))
        
        stack <- mask(stack, m == 0, maskvalue=TRUE)

    }
    for(LC_NAME in MASK_LANDCOVER_NAMES){
        n <- terra::subset(stack, grep('esa_worldcover_v100_2020', names(stack), value=LC_NAME))
        stack <- mask(stack, n == LC_NAME, maskvalue=TRUE)

    }
    rm(m)
}

#read boreal polygon for masking later
boreal_poly <- vect(boreal_vect)

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
                boreal_poly=boreal_poly, 
                predict_var=predict_var,
                max_n=max_n)
