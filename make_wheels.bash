#!/usr/bin/env bash

set -exo pipefail


THISDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd -P )"
md=${1}
shift

PIP_CONFIG_FILE=${PIP_CONFIG_FILE:-${HOME}/.config/pip/pip.conf}
[[ -d $(dirname ${PIP_CONFIG_FILE})  ]]  || mkdir -p $(dirname ${PIP_CONFIG_FILE})
sed "s;PYPI_INDEX_URL;${GITLAB_PYPI}/simple;g" ${THISDIR}/.ci/pip.conf > ${PIP_CONFIG_FILE}
python3 -m pip install -q twine

build-wheels.sh ${md}

if [[ "${md}" != "all" ]] ; then
  python3 -m twine check ${WHEEL_DIR}/final/*${md}*.whl
  python3 -m twine upload --repository-url ${GITLAB_PYPI} ${WHEEL_DIR}/final/*${md}*.whl
fi

echo '************************************'
echo Wheels are in ${WHEEL_DIR}/final
