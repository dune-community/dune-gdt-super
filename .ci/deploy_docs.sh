#!/bin/bash

set -e

ID=${TRAVIS_TAG:-${TRAVIS_BRANCH}}
MODULE=${1}
BUILDDIR=${2}

set -u
pushd /tmp

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
