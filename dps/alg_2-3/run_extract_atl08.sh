#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom
# these libs are NOT included in the base image (vanilla: https://mas.maap-project.org/root/ade-base-images/-/blob/vanilla/docker/Dockerfile)
#conda install -yq -c conda-forge geopandas rio-cogeo rio-tiler importlib_resources
basedir=$( cd "$(dirname "$0")" ; pwd -P )

unset PROJ_LIB

#install requirements packages

pip install --user -r ${basedir}/requirements.txt

mkdir output

FILENAME=$(ls -d input/*)

# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

#command line call that works python /projects/icesat2_boreal/lib/extract_atl08.py -i "/projects/test_data/test_data_30m/ATL08_30m_20181014001049_02350102_003_01.h5" --no-filter-qual --do_30m
python ${basedir}/extract_atl08.py --i $FILENAME --no-filter-qual --do_30m -o output/

