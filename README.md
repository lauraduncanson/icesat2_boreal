# icesat2_boreal
Biomass modeling and mapping in the boreal using NASA's ICESat-2

This repo is developed for a NASA Arctic Boreal Vulnerability Experiment (ABoVE) (PI Laura Duncanson) and ICESat-2 Science Team project (PI Amy Neuenschwander), in collaboration with Paul Montesano, Steve Hancock, Joanne White, Mike Wulder, Eric Guenther, etc.

This is meant to serve as a joint code repository hosting jupyter and/or R notebooks for the following:

1) Reference product creation

1.1 Reference biomass modeling (field:discrete return) - R - LD, JA, TF, PM, CS

1.2 Reference biomass modeling (field:LVIS) - R - (lead TBD) - LD, JA, TF, PM, CS
  - Linking field to LVIS, model development
  - Model options: 1) Field -> aggregated shots over plot, 2) Field -> gridded LVIS

1.3 Reference map production (LVIS) - R - PM, JA, LD, CS
  - LVIS gridding, model application
  
1.4 Estimation and gridding LVIS canopy cover - python - JA, PM
  - following GEDI / Afrisar workflow

1.5 Reference map production (discrete return lidar) - R - LD, JA, TF, PM, CS
  - horizontal reprojection
  

2) ICESat-2 processing

2.1 ICESat-2 data search and download - python - PM, CS, NT, AN, EG

2.2 ICESat-2 subsetting - python - PM, CS, NT, AN, EG
  - icepyx from hackweek could be folded in

2.3 ICESat-2 extraction, filtering - python - AN, PM, LD, TF, JW, SH, JA, EG, CS, NT
  - including filtering using ancillary products, e.g. by cover, landcover
  
2.4 ICESat-2 cover estimation - python - AN, JA, SH, MP, JW, PM

2.5 ICESat-2 30-m ATL08 - python - AN, EG

2.6 ICESat-2 biomass algorithm development - R - LD, JA, CS, AN, SH

2.7 ICESat-2 biomass algorithm application (to full boreal) - CS, NT (PM, LD)


3) Gridded product development

3.1 Create ancillary raster stack - CS, LD, SH (for no snow!), NT

3.2 Develop gridded algorithm - CS, LD, EG, NT

3.3 Apply gridded algorithm to full boreal - CS, NT (PM, LD)


4) Validation

4.1 Comparison of ATL08 heights to ALS heights - EG, AN

4.2 Comparison of ICESat-2 segment biomass estimates to ALS biomass estimates - TBD (NT?)

4.3 Comparison of gridded biomass product to ALS biomass maps - TBD (NT?)

4.4 Comparison of 2019 LVIS to 2010 lidar - TBD

4.5 ICESat-2 Simulator validation - MP, SH, AN

4.5.1 Comparison of ICESat-2 RH metrics to GEDI metrics in the simulator. SH, AN

4.6 Comparison of ICESat-2 and GEDI biomass between 50-52 - R - TBD (LD, JA, AN, PM, SH...)

