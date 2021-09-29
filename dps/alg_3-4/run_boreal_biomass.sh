#!/bin/bash
# this is intended for running DPS jobs; the input directory is where four files have been pulled because download=TRUE in the algorithm_config.yaml file
# a tar file of biomass models, a data table csv, and two raster stack geotiff files

#conda activate r-with-gdal

basedir=$( cd "$(dirname "$0")" ; pwd -P )

unset PROJ_LIB

#install requirements packages - R packages

conda install -c conda-forge -y r-gridExtra r-tidyverse r-randomForest r-raster r-rgdal r-data.table r-rlist r-gdalutils r-stringr r-gdalutils

mkdir output

#unpack biomass models tar
tar -xvf input/bio_models.tar

#FILENAME1=$(ls -d input/*.tar)
FILENAME1=$(ls -d input/*.csv)
FILENAME2=$(ls -d input/C*)
FILENAME3=$(ls -d input/L*)

# Note: the numbered args are fed in with the in_param_dict in the Run DPS chunk of 3.4_dps.ipynb

#TAR_FILE=${4}

# This will put the *rds in the same dir as the R script
#tar -xf ${TAR_FILE} -C ${basedir}

# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

#command line call that works python /projects/icesat2_boreal/lib/extract_atl08.py -i "/projects/test_data/test_data_30m/ATL08_30m_20181014001049_02350102_003_01.h5" --no-filter-qual --do_30m

Rscript mapBoreal.R $FILENAME1 $FILENAME2 $FILENAME3
#Rscript ${basedir}/mapBoreal.R ${1} ${2} ${3}
#gdal_translate output/*.tif -of COG


