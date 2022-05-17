#!/bin/bash
set -x
basedir=$( cd "$(dirname "$0")" ; pwd -P )
#install requirements packages
mamba env create -f  ${basedir}/env_main.yaml
pushd ${HOME}
/opt/conda/envs/icesat2_boreal/bin/pip install --user -e git+https://github.com/MAAP-Project/maap-py.git#egg=maappy

source activate base
mamba install -y -c conda-forge gdal==3.3.3
source activate r-with-gdal
mamba install -y -c conda-forge r-terra
mamba install -y -c conda-forge r-raster
mamba install -y -c conda-forge r-rockchalk==1.8.151

