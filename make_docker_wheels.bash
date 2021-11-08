#!/usr/bin/env bash

ML_TAG=bae2a97ec6937ea8a71ee6f84b096ad783deacf3

THISDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd -P )"

# for rootless/podman execution set both to 0
LOCAL_UID=${LOCAL_UID:-$(id -u)}
LOCAL_GID=${LOCAL_GID:-$(id -g)}
PYTHON_VERSION=${GDT_PYTHON_VERSION:-3.8}

set -eu

IMAGE=zivgitlab.wwu.io/ag-ohlberger/dune-community/docker/manylinux-2014_py${PYTHON_VERSION}:${ML_TAG}
TEST_IMAGE=pymor/testing_py${PYTHON_VERSION}:latest
# check if we a have TTY first, else docker run would throw an error
if [ -t 1 ] ; then 
  DT="-t"
else 
  DT=""
fi

[[ -e ${THISDIR}/docker ]] || mkdir -p ${THISDIR}/docker
export DOCKER_ENVFILE=${THISDIR}/docker/env
python3 ./.ci/shared/scripts/make_env_file.py
# default command is "build-wheels.sh"
# this deletes testtols and uggrid source dirs
docker run ${DT} --env-file=${DOCKER_ENVFILE} -e DUNE_SRC_DIR=/home/dxt/src -v ${THISDIR}:/home/dxt/src \
  -e LOCAL_GID=${LOCAL_GID} -e LOCAL_UID=${LOCAL_UID} -i ${IMAGE}

# makes sure wheels are importable
docker run ${DT} -v ${THISDIR}/docker/wheelhouse/final:/wheelhouse:ro -i ${TEST_IMAGE} \
  bash -c "pip install /wheelhouse/dune* && python -c 'from dune.xt import *; from dune.gdt import *'"

echo '************************************'
echo Wheels are in ${THISDIR}/docker/wheelhouse/final
