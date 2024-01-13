#!/bin/bash

source activate icesat2_boreal
set -x
unset PROJ_LIB

mkdir output

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-3/

FILENAMELIST=($(ls -d input/*))

COVAR_TILE_FN="${PWD}/input/$1"
IN_TILE_FN="${PWD}/input/$2"

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script

# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

#Print to stdout for debugging
python ${basedir}/../../lib/build_stack.py \
--covar_tile_fn ${COVAR_TILE_FN} \
--in_tile_fn ${IN_TILE_FN} \
--in_tile_id_col ${3} \
--in_tile_num ${4} \
--tile_buffer_m ${5} \
--in_tile_layer ${6} \
--output_dir ${OUTPUTDIR} \
--covar_src_name ${7} \
--bandnames_list ${8} \
--in_covar_s3_col ${9} \
--input_nodata_value ${10} \
--shape ${11}
#--clip # not used for topo runs - will be ignored even if this is commented out