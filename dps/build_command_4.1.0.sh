#!/bin/bash
set -x
basedir=$( cd "$(dirname "$0")" ; pwd -P )

source activate pangeo
#install requirements packages
conda env update -f ${basedir}/above_env_3.1.4.yml --solver=libmamba

pushd ${HOME}

# # Do not remove this (PMM Dec 2022)
# source activate icesat2_boreal

# # needed for ee asset export
# conda install --name icesat2_boreal -c conda-forge earthengine-api --solver=libmamba

pip install git+https://github.com/MAAP-Project/maap-py.git#@v4.0.0

# source activate base
# conda install -c conda-forge r-rockchalk
pip3 install pyOpenSSL --upgrade