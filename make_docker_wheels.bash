#!/usr/bin/env bash

ML_TAG=6055ab2

THISDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd -P )"

# for rootless/podman execution set both to 0
LOCAL_UID=${LOCAL_UID:-$(id -u)}
LOCAL_GID=${LOCAL_GID:-$(id -g)}

set -eu
# default command is "build-wheels.sh"
# this deletes testtols and uggrid source dirs
docker run -e DUNE_SRC_DIR=/home/dxt/src -v ${THISDIR}:/home/dxt/src \
  -e LOCAL_GID=${LOCAL_GID} -e LOCAL_UID=${LOCAL_UID} -i dunecommunity/manylinux-2014:${ML_TAG}

# makes sure wheels are importable
docker run -v ${THISDIR}/docker/wheelhouse/final:/wheelhouse:ro -i pymor/testing_py3.8:latest \
  bash -c "pip install /wheelhouse/dune* && python -c 'from dune.xt import *; from dune.gdt import *'"

echo '************************************'
echo Wheels are in ${THISDIR}/docker/wheelhouse/final
