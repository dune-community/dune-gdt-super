#!/bin/bash

set -e

if [ "X${OPTS}" == "X" ]; then
    echo No OPTS specified, defaulting to clang-relwithdebinfo.ninja!
    export OPTS=clang-relwithdebinfo.ninja
fi

# define environment in case we are not in one of our dockers
# determines which one to use below environments/
if [ "X${DXT_ENVIRONMENT}" == "X" ]; then
    echo No environment specified, defaulting to debian-full!
    export DXT_ENVIRONMENT=debian-full
fi

# initialize the virtualenv, if not yet present
export BASEDIR="${PWD}"
mkdir -p environments/${DXT_ENVIRONMENT}/venv
if [ -e environments/${DXT_ENVIRONMENT}/venv/${OPTS} ]; then
  echo not creating virtualenv in environments/${DXT_ENVIRONMENT}/venv/${OPTS}, already exists
else
  cd environments/${DXT_ENVIRONMENT}/venv
  virtualenv --python=python3 ${OPTS}
  source ${OPTS}/bin/activate
  deactivate
fi
cd "${BASEDIR}"
unset BASEDIR

# load the variables of this environment, sources the virtualenv
VENV=environments/${DXT_ENVIRONMENT}/venv/${OPTS} source environments/${DXT_ENVIRONMENT}/PATH.sh

# build dune
if [ "${OPTS: -6}" == ".ninja" ]; then
  MAKE=ninja
else
  MAKE="make -j(($(nproc) - 1))"
fi

cd "${BASEDIR}"
# configure
WERROR=0 ./dune-common/bin/dunecontrol --opts=config.opts/$OPTS --builddir=${BASEDIR}/build-$OPTS configure
for mod in dune-xt dune-gdt; do
  WERROR=1 ./dune-common/bin/dunecontrol --opts=config.opts/$OPTS --only=$mod --builddir=${BASEDIR}/build-$OPTS configure
done
# make all
./dune-common/bin/dunecontrol --builddir=${BASEDIR}/build-$OPTS bexec "$MAKE all"
# make bindings
for mod in dune-xt dune-gdt; do
  ./dune-common/bin/dunecontrol --builddir=${BASEDIR}/build-$OPTS --only=$mod bexec "$MAKE bindings_no_ext"
  ./dune-common/bin/dunecontrol --builddir=${BASEDIR}/build-$OPTS --only=$mod bexec "$MAKE install_python"
done

cd "${BASEDIR}"
echo
echo "All done! From now on run"
echo "  OPTS=$OPTS source envs/${DXT_ENVIRONMENT}/PATH.sh"
echo "to activate the virtualenv!"
