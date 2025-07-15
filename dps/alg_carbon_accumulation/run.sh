#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom

set -x

source activate python

unset PROJ_LIB

mkdir output

# FILENAMELIST=$(ls -d input/*)

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-5/

# First file in input/ dir
# TODO: Fragile relying on alphabetical order

# FILENAMELIST=($(ls -d input/*))
# INPUT1="${PWD}/${FILENAMELIST[0]}"
# INPUT2="${PWD}/${FILENAMELIST[1]}"

## Hard coded args for each run (if any; usually just output dir)

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script

# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

python ${basedir}/../../lib/carbon_accumulation.py \
--map_version ${1} \
--in_tile_num ${2} \
--output_dir ${OUTPUTDIR} \
--n_sims ${3} \
--extent_type ${4} \
--do_write_cog \
--update_age
