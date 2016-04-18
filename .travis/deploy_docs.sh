#!/bin/bash

set -e

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASEDIR=${TRAVIS_BUILD_DIR:-${THISDIR}/..}
ID=${TRAVIS_TAG:-${TRAVIS_BRANCH}}
MODULE=${1}
BUILDDIR=${2}

set -u 
pushd ${BASEDIR}

git clone git@github.com:dune-community/dune-community.github.io.git site

cd site
git config user.name "DUNE Community Bot"
git config user.email "dune-community.bot@wwu.de"

TARGET=docs/${MODULE}/${ID}/
mkdir -p ${TARGET}
rsync -a --delete  ${BUILDDIR}/${MODULE}/doc/doxygen/html/ ${TARGET}/
git add ${TARGET}

git commit -m "Updated documentation for ${MODULE} ${ID}"
git push

popd
