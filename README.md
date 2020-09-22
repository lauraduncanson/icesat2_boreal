# icesat2_boreal
Biomass modeling and mapping of forest biomass in the boreal using NASA's ICESat-2

This repo is developed for a NASA Arctic Boreal Vulnerability Experiment (ABoVE) (PI Laura Duncanson) and ICESat-2 Science Team project (PI Amy Neuenschwander), in collaboration with Paul Montesano, Steve Hancock, Joanne White, Mike Wulder, Eric Guenther, etc.

This is meant to serve as a joint code repository hosting jupyter and/or R notebooks for the following:

1) Reference product creation

1.1 Reference biomass modeling (field:discrete return) - LD, JA, TF, PM

1.2 Reference biomass modeling (field:LVIS) - LD, JA, TF, PM

1.3 Reference map production (LVIS) - LD, JA, TF, PM

1.4 Reference map production (discrete return lidar) - LD, JA, TF, PM

2) ICESat-2 processing

2.1 ICESat-2 data search and download - PM

2.2 ICESat-2 extraction, filtering + cover? - AN, PM, LD, TF, JW, SH, JA, EG

2.3 ICESat-2 biomass algorithm development (including subsetting?) - LD, AR, AN

2.4 ICESat-2 biomass algorithm application (to full boreal) - AR (PM, LD)

3) Gridded product development

3.1 Create ancillary raster stack - AR

3.2 Develop gridded algorithm - TBD

3.3 Apply gridded algorithm to full boreal - AR (PM, LD)


4) Validation

4.1 Comparison of ATL08 heights to ALS heights - EG

4.2 Comparison of ICESat-2 segment biomass estimates to ALS biomass estimates - TBD (AR)

4.3 Comparison of gridded biomass product to ALS biomass maps - TBD (AR)

4.4 Comparison of 2019 LVIS to 2010 lidar - TBD

4.5 ICESat-2 Simulator validation - MP, SH, AN

4.5.1 Comparison of ICESat-2 RH metrics to GEDI metrics in the simulator. SH, AN

