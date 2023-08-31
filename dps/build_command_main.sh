#!/bin/bash
set -x
basedir=$( cd "$(dirname "$0")" ; pwd -P )

#install requirements packages
mamba env update -f ${basedir}/above_env.yml

# needed for ee asset export
mamba install -c conda-forge earthengine-api

pushd ${HOME}

# Do not remove this (PMM Dec 2022)
# Testing remove (PMM Aug 2023)
#source activate icesat2_boreal

pip install git+https://github.com/MAAP-Project/maap-py.git#egg=maappy

source activate base
pip3 install pyOpenSSL --upgrade