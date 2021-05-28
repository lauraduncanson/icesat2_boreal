#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom
# these libs are NOT included in the base image (vanilla: https://mas.maap-project.org/root/ade-base-images/-/blob/vanilla/docker/Dockerfile)
#conda install -yq -c conda-forge geopandas rio-cogeo rio-tiler importlib_resources
unset PROJ_LIB
pip install -U geopandas rasterio rio-cogeo rio-tiler

mkdir output

FILENAMELIST=$(ls -d input/*)

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-5/

# First file in input/ dir
# TODO: Fragile relying on alphabetical order
#python get_param.py in_tile_fn
INPUT1="${PWD}/${FILENAMELIST[0]}"
INPUT2="${PWD}/${FILENAMELIST[1]}"

#FILELIST=($INPUTFILE)

## Hard coded args for each run (if any; usually just output dir)

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script

# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

#cd ${basedir}/notebooks/3.Gridded product development/

#TODO: 
#python setup.py install -prefix=${basedir}/gedipy_bin

#export PYTHONPATH="${PYTHONPATH}:${basedir}/notebooks/3.Gridded product development/"

# Cmd line call that worked
#python 3.1.2_dps.py -i /projects/maap-users/alexdevseed/boreal_tiles.gpkg -n 30543 -l boreal_tiles_albers  -o /projects/tmp/Landsat/ -b 0 --json_path /projects/maap-users/alexdevseed/landsat8/sample2/

python ${basedir}/../../notebooks/3.Gridded_product_development/3.1.2_dps.py \
--in_tile_fn ${INPUT1} \
--in_tile_num $1 \
--in_tile_layer $2 \
--sat_api $3 \
--local $4 \
--output_dir ${OUTPUTDIR}