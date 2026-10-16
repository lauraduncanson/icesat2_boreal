#!/bin/bash
# this is intended for running DPS jobs

# Useful to explicitly source the env you created in the build command right here
#source activate icesat2_boreal
#source activate pangeo
source activate python

set -x
basedir=$( cd "$(dirname "$0")" ; pwd -P )

unset PROJ_LIB

mkdir output

# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

# ${3} (optional): band renames "old1:new1,old2:new2" (no spaces); empty = automatic S1 naming only
BAND_RENAME_ARG=""
if [ -n "${3}" ]; then
    BAND_RENAME_ARG="-band_rename ${3}"
fi

python ${basedir}/../../lib/extract_atl08_covars.py \
-s3_atl08_gdf_fn ${1} \
-tindex_fn_list ${2} \
-outdir ${OUTPUTDIR} ${BAND_RENAME_ARG}