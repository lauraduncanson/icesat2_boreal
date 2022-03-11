#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom
# these libs are NOT included in the base image (vanilla: https://mas.maap-project.org/root/ade-base-images/-/blob/vanilla/docker/Dockerfile)
#conda install -yq -c conda-forge geopandas rio-cogeo rio-tiler importlib_resources
basedir=$( cd "$(dirname "$0")" ; pwd -P )

unset PROJ_LIB

#install requirements packages

#pip install --user -r ${basedir}/requirements.txt

mkdir output

FILENAMELIST=($(ls -d input/*gpkg))
INPUT1="${PWD}/${FILENAMELIST[0]}"
#INPUT2="${PWD}/${FILENAMELIST[1]}

# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

cmd="python ${basedir}/../../lib/tile_atl08.py \
--updated_filters \
--extract_covars \
--do_dps \
--do_30m \
-o ${OUTPUTDIR} \
-in_tile_num ${1} \
-in_tile_fn ${INPUT1} \
-in_tile_layer ${2} \
-csv_list_fn ${3} \
-topo_stack_list_fn ${4} \
-landsat_stack_list_fn ${5} \
-years_list ${6} \
-user_stacks ${7} \
-user_atl08 ${8} \
-thresh_sol_el ${9} \
-v_ATL08 ${10} \
-minmonth ${11} \
-maxmonth ${12}"

# Print the command to stdout for debugging
echo ${cmd}
# Run the command
eval ${cmd}
