#!/bin/bash

source activate python

set -x
unset PROJ_LIB

mkdir output

basedir=$( cd "$(dirname "$0")" ; pwd -P )  

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script


# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

python ${basedir}/../../lib/tile_forestage.py \
--in_url ${1} \
--in_vector_fn ${2} \
--in_id_col ${3} \
--in_id_num ${4} \
--year ${5} \
--no_data_val ${6} \
--output_dir ${OUTPUTDIR}