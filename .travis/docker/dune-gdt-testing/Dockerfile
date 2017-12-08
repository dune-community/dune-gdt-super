FROM dunecommunity/testing-base:latest

# COMPILER Selection (ARGS cannot be uppercase)
ARG cc
ARG cxx
ARG commit
ENV CC=$cc \
    CXX=$cxx \
    MY_MODULE=dune-gdt

WORKDIR ${SUPERDIR}
RUN ${DUNE_VENV_ACTIVATE} && \
    git clone https://github.com/dune-community/dune-gdt-super.git ${SUPERDIR} && \
    git checkout ${commit} && \
    cd ${SUPERDIR} && \
    git submodule foreach git reset --hard && \
    git submodule update --init --recursive && \
    ./bin/download_external_libraries.py && \
    ./bin/build_external_libraries.py && \
    rm -rf ${MY_MODULE} && \
    mkdir ${MY_MODULE} && \
    ${SRC_DCTRL} ${BLD} all

# THE END
ENV DEBIAN_FRONTEND teletype
