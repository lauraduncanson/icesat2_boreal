#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom

set -x

source activate python

unset PROJ_LIB

mkdir output

FILENAMELIST=$(ls -d input/*)

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-5/

# First file in input/ dir
# TODO: Fragile relying on alphabetical order

FILENAMELIST=($(ls -d input/*))
INPUT1="${PWD}/${FILENAMELIST[0]}"
INPUT2="${PWD}/${FILENAMELIST[1]}"

## Hard coded args for each run (if any; usually just output dir)

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script

# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

python ${basedir}/../../lib/build_ms_composite.py \
--in_tile_fn ${INPUT1} \
--in_tile_num ${1} \
--in_tile_layer ${2} \
--sat_api ${3} \
--tile_buffer_m ${4} \
--start_year ${5} \
--end_year ${6} \
--start_month_day ${7} \
--end_month_day ${8} \
--max_cloud ${9} \
--composite_type ${10} \
--shape ${11} \
--hls_product ${12} \
--thresh_min_ndvi ${13} \
--min_n_filt_results ${14} \
--stat ${15} \
--stat_pct ${16} \
--target_spectral_index ${17} \
--output_dir ${OUTPUTDIR}
