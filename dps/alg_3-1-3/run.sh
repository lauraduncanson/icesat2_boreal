#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file
#conda activate r-with-gdal
# This installs the python libs needed to run the script at the bottom
# these libs are NOT included in the base image (vanilla: https://mas.maap-project.org/root/ade-base-images/-/blob/vanilla/docker/Dockerfile)
#conda install -yq -c conda-forge geopandas rio-cogeo rio-tiler importlib_resources
#pip install --user -U numpy==1.20.3 geopandas rio-cogeo rio-tiler==2.0.8 rasterio==1.2.6 importlib_resources

source activate icesat2_boreal
set -x
unset PROJ_LIB

mkdir output

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-3/

#pip install --user -U ${basedir}/../requirements_main.txt

FILENAMELIST=($(ls -d input/*))

#### How to make sure INPUT1 if the 'in_tile_fn' and INPUT2 is the covar_tile_fn???
COVAR_TILE_FN="${PWD}/input/$1"
IN_TILE_FN="${PWD}/input/$2"

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script

# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

# Cmd line call that worked
#python /projects/Developer/icesat2_boreal/lib/build_stack.py --in_tile_fn /projects/my-public-bucket/boreal_tiles_v003.gpkg --in_tile_id_col tile_num --in_tile_num 3131 --tile_buffer_m 0 --in_tile_layer boreal_tiles_v003 -o /projects/my-public-bucket/DPS_ESA_LC --topo_off --covar_src_name esa_worldcover_v100_2020 --covar_tile_fn /projects/my-public-bucket/analyze_agb/footprints_v100_2020_v100_2020_map-s3.gpkg --in_covar_s3_col s3_path --input_nodata_value 0 --clip

#Print to stdout for debugging
python ${basedir}/../../lib/build_stack.py \
--covar_tile_fn ${COVAR_TILE_FN} \
--in_tile_fn ${IN_TILE_FN} \
--in_tile_id_col $3 \
--in_tile_num $4 \
--tile_buffer_m $5 \
--in_tile_layer $6 \
--output_dir $OUTPUTDIR \
--topo_off \
--covar_src_name $7 \
--in_covar_s3_col $8 \
--input_nodata_value $9 \
--clip