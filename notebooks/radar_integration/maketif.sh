#!/bin/bash
INPUT=$1
OUTPUT="$(basename -- $INPUT)"
OUTPATH="/projects/my-public-bucket/sentinel1_seasonal_comps/GEE/large_test_tiffs/"
echo rio cogeo create "${INPUT}" "${OUTPATH}${OUTPUT}.tif"
rio cogeo create "${INPUT}" "${OUTPATH}${OUTPUT}.tif"