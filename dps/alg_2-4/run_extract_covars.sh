#!/bin/bash
# this is intended for running DPS jobs

source activate icesat2_boreal
set -x
basedir=$( cd "$(dirname "$0")" ; pwd -P )

unset PROJ_LIB

mkdir output

# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

python ${basedir}/../../lib/extract_covars.py \
-s3_atl08_gdf_fn ${1} \
-tindex_fn_list ${2}