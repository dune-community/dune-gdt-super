FROM dunecommunity/testing-base:master

# COMPILER Selection (ARGS cannot be uppercase)
ARG cc
ARG cxx
ARG branch
ENV CC=$cc \
    CXX=$cxx \
    MY_MODULE=dune-gdt

WORKDIR ${SUPERDIR}
RUN ${DUNE_VENV_ACTIVATE} && \
    git clone --branch=${branch} https://github.com/dune-community/dune-gdt-super.git ${SUPERDIR} && \
    cd ${SUPERDIR} && \
    git submodule foreach git reset --hard && \
    git submodule update --init --recursive && \
    ./local/bin/download_external_libraries.py && \
    ./local/bin/build_external_libraries.py && \
    rm -rf ${MY_MODULE} && \
    mkdir ${MY_MODULE} && \
    ${SRC_DCTRL} ${BLD} all

# THE END
ENV DEBIAN_FRONTEND teletype
