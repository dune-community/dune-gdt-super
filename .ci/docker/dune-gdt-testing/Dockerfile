FROM dunecommunity/testing-base_debian:latest

# COMPILER Selection (ARGS cannot be uppercase)
ARG cc
ARG cxx
ARG commit
ENV CC=$cc \
    CXX=$cxx \
    MY_MODULE=dune-gdt

RUN git clone https://github.com/dune-community/dune-gdt-super.git ${SUPERDIR} && \
    cd ${SUPERDIR} && \
    git checkout ${commit} && \
    git submodule foreach git reset --hard && \
    git submodule update --init --recursive && \
    ./bin/download_external_libraries.py && \
    ./bin/build_external_libraries.py && \
    ${SRC_DCTRL} ${BLD} all && \
    rm -rf ${MY_MODULE} && \
    mkdir ${MY_MODULE}

VOLUME ${SUPERDIR}/${MY_MODULE}
