#!/bin/bash --login

basedir=$( cd "$(dirname "$0")" ; pwd -P )
#install requirements packages
pip install --user -U -r ${basedir}/requirements_main.txt

