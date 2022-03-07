#!/usr/bin/env bash

set -exo pipefail


THISDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd -P )"
md=${1}
shift

PIP_CONFIG_FILE=${PIP_CONFIG_FILE:-${HOME}/.config/pip/pip.conf}
[[ -d $(dirname ${PIP_CONFIG_FILE})  ]]  || mkdir -p $(dirname ${PIP_CONFIG_FILE})
sed "s;PYPI_INDEX_URL;${GITLAB_PYPI}/simple;g" ${THISDIR}/.ci/pip.conf > ${PIP_CONFIG_FILE}
python3 -m pip install -q twine

set +eu
# check if the to-be-build version is already installable and exit early
# this check requires completed `dunecontrol all` 
if [[ "${md}" != "all" ]] ; then
    export MD_VERSION=$(cat ${DUNE_BUILD_DIR}/dune-xt/python/version.sh)
    python3 -m pip install dune-${md}==${MD_VERSION}
    if [[ $? -ne 0 ]]; then
        echo "Already built dune-${md}==${MD_VERSION}, skipping" 
        exit 0
    fi
fi
set -eu

build-wheels.sh ${md}

echo '************************************'
echo Wheels are in ${WHEEL_DIR}/final
