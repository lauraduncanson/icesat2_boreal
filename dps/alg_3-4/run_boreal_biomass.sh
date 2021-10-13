#!/bin/bash --login
# this is intended for running DPS jobs; the input directory is where four files have been pulled because download=TRUE in the algorithm_config.yaml file
# a tar file of biomass models, a data table csv, and two raster stack geotiff files

conda activate r-with-gdal

basedir=$( cd "$(dirname "$0")" ; pwd -P )

unset PROJ_LIB

pip install --user -r ${basedir}/requirements.txt

#install requirements packages - R packages

#conda install -c conda-forge -y r-gridExtra r-tidyverse r-randomForest r-raster r-rgdal r-data.table r-rlist r-gdalutils r-stringr r-gdalutils

mkdir output

# Note: the numbered args are fed in with the in_param_dict in the Run DPS chunk of 3.4_dps.ipynb

ATL08_CSV=${1}
TOPO_TIF=${2}
LANDSAT_TIF=${3}
${DO_SLOPE_VALID_MASK}=${4}
ATL08_SAMPLE_CSV=${5}
in_tile_num=${6}
in_tile_fn=${7}
in_tile_layer=${8}


TAR_FILE=${basedir}/bio_models.tar


#unpack biomass models tar
#tar -xvf input/bio_models.tar

# This will put the *rds in the same dir as the R script
tar -xf ${TAR_FILE}

# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

# Get the output merged CSV of filtered ATL08 for the input tile and its neighbors
cmd="python ${basedir}/merge_neighbors_atl08.py -in_tile_num ${in_tile_fn} -in_tile_fn ${in_tile_fn} -in_tile_field ${in_tile_layer} -csv_list_fn ${ATL08_CSV} -out_dir ${OUTPUTDIR}"
echo $cmd
eval $cmd

# Set the output merged CSV name to a var
MERGED_ATL08_CSV=$(ls ${OUTPUTDIR}/atl08_004_30m_filt_merge_neighbors* | head -1)

# Run mapBoreal with merged CSV as input
cmd="Rscript ${basedir}/mapBoreal.R ${MERGED_ATL08_CSV} ${TOPO_TIF} ${LANDSAT_TIF} ${DO_SLOPE_VALID_MASK} ${ATL08_SAMPLE_CSV}"

echo $cmd
eval $cmd


