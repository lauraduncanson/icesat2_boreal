#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file
# only other arg for now is the output resolution

unset PROJ_LIB

mkdir output

basedir=$( cd "$(dirname "$0")" ; pwd -P ) 

FILENAMELIST=($(ls -d input/*.tif))
INPUT1="${PWD}/${FILENAMELIST[0]}"
OUTPUT="${INPUT1/.tif/_${1}m.tif}" 

# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"


#Print to stdout for debugging
echo gdal_translate -of COG -tr $1 $1 -co COMPRESS=DEFLATE -co PREDICTOR=2 ${INPUT1} "${OUTPUTDIR}/${OUTPUT}"

