# icesat2_boreal

Biomass modeling and mapping in the boreal using NASA's ICESat-2, Harmonized Sentinel Landsat (HLS) and the Copernicus DEM

This repo is developed for a NASA Arctic Boreal Vulnerability Experiment (ABoVE) (PI Laura Duncanson) and ICESat-2 Science Team project (PI Amy Neuenschwander), in collaboration with Paul Montesano, Nathan Thomas, Carlos Silva, Steve Hancock, Joanne White, Mike Wulder, Eric Guenther, and many others.

This is meant to serve as a joint code repository hosting jupyter and/or R notebooks for the following (notionally):

1) Reference product creation

1.1 Reference biomass modeling (field:discrete return)

1.2 Reference biomass modeling (field:LVIS) - R
  - Linking field to LVIS, model development
  - Model options: 1) Field -> aggregated shots over plot, 2) Field -> gridded LVIS

1.3 Reference map production (LVIS)
  - LVIS gridding, model application

1.5 Reference map production (discrete return lidar)
  - horizontal reprojection
  

2) ICESat-2 processing

2.1 ICESat-2 data search and download

2.2 ICESat-2 subsetting

2.3 ICESat-2 extraction, filtering
  - including filtering using ancillary products, e.g. by cover, landcover
  
2.5 ICESat-2 30-m ATL08
   - inseparate repo by Eric Guenther


3) Gridded product development

3.1 Create ancillary raster stack

3.2 Develop gridded algorithm

3.4 Apply icesat-2 biomass models and apply gridded algorithm to full boreal


4) Validation

4.1 Comparison of ATL08 heights to LVIS heights 

4.2 Comparison of ICESat-2 segment biomass estimates to ALS biomass estimates 

4.3 Comparison of gridded biomass product to ALS biomass maps 

4.4 Comparison of 2019 LVIS to 2010 lidar - coming soon...

4.6 Comparison of ICESat-2 and GEDI biomass between 50-52 - coming soon...
