#!/usr/bin/env bash

THISDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd -P )"

set -e
# default command is "build-wheels.sh"
# this deletes testtols and uggrid source dirs
docker run -e DUNE_SRC_DIR=/home/dxt/src -v ${THISDIR}:/home/dxt/src \
  -it dunecommunity/manylinux-2014:ddb4b80

docker run -v ${THISDIR}/docker/wheelhouse/final:/wheelhouse:ro -it pymor/testing_py3.8:latest \
  bash -c "pip install /wheelhouse/dune* && python -c 'from dune.xt import *; from dune.gdt import *'"

echo '************************************'
echo Wheels are in ${THISDIR}/docker/wheelhouse/final
