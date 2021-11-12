#!/usr/bin/env bash

ML_TAG=45cc22db07484489e5cf9a7459bdbe32f4567ad2

THISDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd -P )"

# for rootless/podman execution set both to 0
LOCAL_USER=${LOCAL_USER:$USER}
LOCAL_UID=${LOCAL_UID:-$(id -u)}
LOCAL_GID=${LOCAL_GID:-$(id -g)}
EXEC=${EXEC}
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
  -e LOCAL_USER=${LOCAL_USER} -e LOCAL_GID=${LOCAL_GID} -e LOCAL_UID=${LOCAL_UID} -i ${IMAGE} ${EXEC}

# makes sure wheels are importable
docker run ${DT} -v ${THISDIR}/docker/wheelhouse/final:/wheelhouse:ro -i ${TEST_IMAGE} \
  bash -c "pip install /wheelhouse/dune* && python -c 'from dune.xt import *; from dune.gdt import *'"

echo '************************************'
echo Wheels are in ${THISDIR}/docker/wheelhouse/final
