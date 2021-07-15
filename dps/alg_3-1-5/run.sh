#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom
# these libs are NOT included in the base image (vanilla: https://mas.maap-project.org/root/ade-base-images/-/blob/vanilla/docker/Dockerfile)
#conda install -yq -c conda-forge geopandas rio-cogeo rio-tiler importlib_resources
pip install --user -U numpy==1.20.3 geopandas rio-cogeo rio-tiler==2.0.8 rasterio==1.2.6 importlib_resources
#pip install --user -U /projects/requirements.txt

mkdir output

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-5/

# First file in input/ dir
# TODO: Fragile relying on alphabetical order
#python get_param.py in_tile_fn
FILENAMELIST=($(ls -d input/*))
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
#python 3.1.5_dps.py --in_tile_fn '/projects/maap-users/alexdevseed/boreal_tiles.gpkg' --in_tile_num 30550 --tile_buffer_m 120 --in_tile_layer "boreal_tiles_albers" -o '/projects/tmp/Topo/'

#Print to stdout for debugging
echo python ${basedir}/../../notebooks/3.Gridded_product_development/3.1.5_dps.py \
--in_tile_fn ${INPUT1} \
--in_tile_num $1 \
--tile_buffer_m $2 \
--in_tile_layer $3 \
--output_dir $OUTPUTDIR \
--tmp_out_path $OUTPUTDIR \
--topo_tile_fn ${INPUT2}

python ${basedir}/../../notebooks/3.Gridded_product_development/3.1.5_dps.py \
--in_tile_fn ${INPUT1} \
--in_tile_num $1 \
--tile_buffer_m $2 \
--in_tile_layer $3 \
--output_dir $OUTPUTDIR \
--tmp_out_path $OUTPUTDIR \
--topo_tile_fn ${INPUT2}