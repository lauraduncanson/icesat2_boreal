#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom

set -x

source activate python

unset PROJ_LIB

mkdir output

# FILENAMELIST=$(ls -d input/*)

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-5/

## Hard coded args for each run (if any; usually just output dir)

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script

# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

python ${basedir}/../../lib/compute_trends.py \
--value-rasters ${1} \
--date-rasters ${2} \
--output ${3} \
--outdir ${OUTPUTDIR} \
--alpha ${4} \
--n-processes ${5} \
--chunk-size ${6} \
--do-ols \
--no-breakpoints \
--verbose
