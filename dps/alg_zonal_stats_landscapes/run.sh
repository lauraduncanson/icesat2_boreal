#!/bin/bash
# this is intended for running DPS jobs; the input directory is where a single file has been pulled because download=TRUE in the algorithm_config.yaml file

# This installs the python libs needed to run the script at the bottom

set -x

source activate python

unset PROJ_LIB

mkdir output

basedir=$( cd "$(dirname "$0")" ; pwd -P )  # goes to alg_3-1-5/

## Hard coded args for each run (if any; usually just output dir)

# Work dir is always from where your script is called
# Base dir is always the relative dir within the run*.sh script

# Absolute path here
# This PWD is wherever the job is run (where the .sh is called from) 
OUTPUTDIR="${PWD}/output"

# This zonal stats returns a biomass-specific monte carlo summary that includes a sum (in Pg)
# Pairing with calculate_biomass_change.py formats the summary for age x trend class combos 
python ${basedir}/../../lib/zonal_stats_landscapes.py \
--polygons ${1} \
--polygon-id-col ${2} \
--polygon-ids ${3} \
--output  ${OUTPUTDIR}/biomass_results.gpkg \
--mosaic-geojsons ${4} \
--zs-col-prefix ${5} \
--bands ${6} \
--statistics ${7} \
--age-classes ${8} \
--age-mosaic-index ${9} \
--age-band ${10} \
--trend-mosaic-index ${11} \
--trend-band ${12} \
--additional-biomass-indices ${13} \
--additional-biomass-bands ${14} \
--additional-biomass-names ${15} \
--processes ${16} \
--chunk-size ${17}

python ${basedir}/../../lib/calculate_biomass_change.py \
--input  ${OUTPUTDIR}/biomass_results_polygon_summary.csv \
--output  ${OUTPUTDIR}/biomass_results_polygon_biomass_change.csv \
--summary-output ${OUTPUTDIR}/biomass_results_change_summary.csv \
--year1-prefix ${18} \
--year2-prefix ${19} \
--simulations ${20}

# This zonal stats returns a monte carlo results summary of the 'uncertainty pair', which here is carbon accumulation 2020, for age x trend class combos
python ${basedir}/../../lib/zonal_stats_landscapes_misc.py \
--polygons ${1} \
--polygon-id-col ${2} \
--polygon-ids ${3} \
--output  ${OUTPUTDIR}/cacc_results.gpkg \
--mosaic-geojsons ${4} \
--zs-col-prefix ${5} \
--bands ${6} \
--uncertainty-pairs ${21} \
--age-classes ${8} \
--age-mosaic-index ${9} \
--age-band ${10} \
--trend-mosaic-index ${11} \
--trend-band ${12} \
--processes ${16} \
--chunk-size ${17}