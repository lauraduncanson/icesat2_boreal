#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom
# these libs are NOT included in the base image (vanilla: https://mas.maap-project.org/root/ade-base-images/-/blob/vanilla/docker/Dockerfile)
#conda install -yq -c conda-forge geopandas rio-cogeo rio-tiler importlib_resources
set -x
source activate icesat2_boreal

unset PROJ_LIB
#conda list
#pip install --user -U numpy==1.20.3 geopandas==0.9.0 rio-cogeo==2.3.1 rio-tiler==2.1.4 rasterio==1.2.6 morecantile==2.1.4 pystac-client importlib_resources 

mkdir output

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-5/

# First file in input/ dir
# TODO: Fragile relying on alphabetical order

# DPS notebook used to submit this job will make a new credentials file
# The notebook will authenticate a new creds file and copy to my-private-bucket/GEE/credentials
# In the yaml for this alg, there is a 'required file' passed to submitJob that downloads this creds file using the supplied s3 path.
# Then, this bash script can access the file like this:
INPUT1=${PWD}/input/credentials
gpkg_files=(${PWD}/input/*.gpkg)
ASSET_GDF_FN=${gpkg_files[0]}

# Move to this dir...
mkdir -p ${PWD}/input/.config/earthengine/
cp ${INPUT1} ${PWD}/input/.config/earthengine/

# Also, move cred to this dir...
mkdir -p ${PWD}/.config/earthengine/
cp ${INPUT1} ${PWD}/.config/earthengine/

# Or maybe move here?
mkdir -p .config/earthengine/
cp ${INPUT1} .config/earthengine/

## Or maybe here?
#mkdir -p /root/.config/earthengine/
#chmod 644 /root/.config/earthengine/
#cp ${INPUT1} /root/.config/earthengine/
#chmod 644 /root/.config/earthengine/credentials
#
#echo "Check if creds file is readable"
#ls -lht /root/.config/earthengine/*

# HOME directory is getting set to /root by the metrics collector script causing the creds file to not be found (sujen)
# so, do this
export HOME=/home/ops
mkdir -p $HOME/.config/earthengine/
#chmod 644 $HOME/.config/earthengine/
cp ${INPUT1} $HOME/.config/earthengine/
chmod 644 $HOME/.config/earthengine/credentials

echo "Check if creds file is readable"
ls -lht $HOME/.config/earthengine/*

## Hard coded args for each run (if any; usually just output dir)

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script

# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

python ${basedir}/../../lib/do_gee_download_by_subtile.py \
--subtile_loc ${1} \
--id_num ${2} \
--id_col ${3} \
--tile_size_m ${4} \
--asset_path ${5} \
--asset_gdf_fn ${ASSET_GDF_FN} \
--out_dir ${OUTPUTDIR}
