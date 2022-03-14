#!/bin/bash --login

basedir=$( cd "$(dirname "$0")" ; pwd -P )
#install requirements packages
/opt/conda/envs/r-with-gdal/bin/pip install --user -U -r ${basedir}/requirements_main.txt
echo "conda activate r-with-gdal" >> ~/.bashrc
