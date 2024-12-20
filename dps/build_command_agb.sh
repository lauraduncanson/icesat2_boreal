#!/bin/bash
set -x
basedir=$( cd "$(dirname "$0")" ; pwd -P )

#install requirements packages
conda env update -f ${basedir}/above_env_r_3.1.4.yml

pushd ${HOME}

# Do not remove this (PMM Dec 2022)
source activate icesat2_boreal

# needed for ee asset export
#mamba install --name icesat2_boreal -c conda-forge earthengine-api

pip install git+https://github.com/MAAP-Project/maap-py.git#egg=maappy

source activate r
pip3 install pyOpenSSL --upgrade