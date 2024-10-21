#!/bin/bash
set -x
basedir=$( cd "$(dirname "$0")" ; pwd -P )

# This is designed to update the base image called 'python'
# https://github.com/MAAP-Project/maap-workspaces/blob/main/base_images/python/environment.yml
conda env update -f ${basedir}/above_env_4.1.0.yml --solver=libmamba

# New name of the conda env created in above_env_4.1.0.yml
source activate python

pushd ${HOME}