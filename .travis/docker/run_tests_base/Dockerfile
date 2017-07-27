FROM dunecommunity/testing-base:latest

# COMPILER Selection (ARGS cannot be uppercase)
ARG cc
ARG cxx
ENV CC=$cc \
    CXX=$cxx \
    MY_MODULE=dune-gdt \
    TRAVIS=1

WORKDIR ${SUPERDIR}
ADD . ${SUPERDIR}/
RUN ${DUNE_VENV_ACTIVATE} && \
    ./local/bin/download_external_libraries.py && \
    ./local/bin/build_external_libraries.py && \
    ${SRC_DCTRL} ${BLD} all
