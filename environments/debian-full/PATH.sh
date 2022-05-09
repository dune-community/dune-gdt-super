BASEDIR="$(cd "$(dirname ${BASH_SOURCE[0]})/../.." ;  pwd -P )"
export DXT_ENVIRONMENT=${DXT_ENVIRONMENT:-debian-full}
export OPTS=${OPTS:-clang-relwithdebinfo.ninja}
export VENV=${VENV:-environments/${DXT_ENVIRONMENT}/venv/${OPTS}}
export PATH=${BASEDIR}/bin:$PATH
export F77=gfortran
export CMAKE_FLAGS="-DDS_HEADERCHECK_DISABLE=1 -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE -DMINIMAL_DEBUG_LEVEL=grave -DDUNE_PYTHON_VIRTUALENV_SETUP=1 -DDUNE_PYTHON_VIRTUALENV_PATH=${BASEDIR}/${VENV}"
if [ -e "${BASEDIR}/${VENV}/bin/activate" ]; then
    source "${BASEDIR}/${VENV}/bin/activate"
else
    echo "WARNING: missing virtualenv ${BASEDIR}/${VENV}, did you call build_interactive_bindings_venv.sh?"
fi
export OMP_NUM_THREADS=1
