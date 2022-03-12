#!/bin/bash --login

basedir=$( cd "$(dirname "$0")" ; pwd -P )
conda activate r-with-gdal
#install requirements packages
pip install --user -U -r ${basedir}/requirements_main.txt

