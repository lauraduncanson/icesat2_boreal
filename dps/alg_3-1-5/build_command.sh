#!/bin/bash --login

basedir=$( cd "$(dirname "$0")" ; pwd -P )
#install requirements packages
pip install --user -U numpy==1.20.3 geopandas rio-cogeo rio-tiler==2.0.8 rasterio==1.2.6 importlib_resources