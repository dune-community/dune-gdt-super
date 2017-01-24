FROM dunecommunity/testing-base:master

# COMPILER Selection (ARGS cannot be uppercase)
ARG cc
ARG cxx
ENV CC=$cc \
    CXX=$cxx \
    MY_MODULE=dune-gdt \
    PATH=/usr/lib/ccache:${PATH} \
    OPTS=config.opts/travis.ninja \
    SRC_DCTRL=${SUPERDIR}/dune-common/bin/dunecontrol

WORKDIR ${SUPERDIR}
ADD . ${SUPERDIR}

RUN cd ${SUPERDIR} && \
    git submodule foreach git reset --hard && \
    git submodule update --init --recursive && \
    git submodule status && \
    ./local/bin/download_external_libraries.py && \
    ./local/bin/build_external_libraries.py && \
    rm -rf ${MY_MODULE} && \
    mkdir ${MY_MODULE} && \
    ${SRC_DCTRL} ${BLD} all

# THE END
ENV DEBIAN_FRONTEND teletype
